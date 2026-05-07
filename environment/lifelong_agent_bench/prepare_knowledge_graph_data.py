from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LifelongAgentBench knowledge_graph parquet into the local official layout."
    )
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--entry-dict-out", type=Path, required=True)
    parser.add_argument("--ontology-dir-out", type=Path, required=True)
    return parser.parse_args()


def _parse_literal(value: str):
    return ast.literal_eval(value) if isinstance(value, str) else value


def _extract_relation(action: str) -> str | None:
    if not action.startswith("get_neighbors(") or not action.endswith(")"):
        return None
    inner = action[len("get_neighbors(") : -1]
    parts = inner.split(",", 1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


def _extract_attribute(action: str) -> str | None:
    if not action.endswith(")") or "(" not in action:
        return None
    if not (action.startswith("argmax(") or action.startswith("argmin(")):
        return None
    inner = action[action.index("(") + 1 : -1]
    parts = inner.split(",", 1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.parquet)
    rows = table.to_pylist()

    entry_dict: dict[str, dict] = {}
    relation_set: set[str] = set()
    attribute_set: set[str] = set()

    for row in rows:
        sample_index = str(row["sample_index"])
        entity_dict = _parse_literal(row["entity_dict"])
        action_list = _parse_literal(row["action_list"])
        answer_list = _parse_literal(row["answer_list"])
        skill_list = _parse_literal(row["skill_list"])

        for action in action_list:
            relation = _extract_relation(action)
            if relation:
                relation_set.add(relation)
            attribute = _extract_attribute(action)
            if attribute:
                attribute_set.add(attribute)

        entry_dict[sample_index] = {
            "question": row["question"],
            "qid": row["qid"],
            "source": row["source"],
            "entity_dict": entity_dict,
            "s_expression": row["s_expression"],
            "action_list": action_list,
            "answer_list": answer_list,
            "skill_list": skill_list,
        }

    args.entry_dict_out.parent.mkdir(parents=True, exist_ok=True)
    args.ontology_dir_out.mkdir(parents=True, exist_ok=True)

    args.entry_dict_out.write_text(
        json.dumps(entry_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.ontology_dir_out / "vocab.json").write_text(
        json.dumps(
            {
                "attributes": sorted(attribute_set),
                "relations": sorted(relation_set),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.ontology_dir_out / "fb_roles").write_text(
        "".join(f"type.object {relation} type.object\n" for relation in sorted(relation_set)),
        encoding="utf-8",
    )
    (args.ontology_dir_out / "reverse_properties").write_text("", encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": len(entry_dict),
                "relations": len(relation_set),
                "attributes": len(attribute_set),
                "entry_dict_out": str(args.entry_dict_out),
                "ontology_dir_out": str(args.ontology_dir_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
