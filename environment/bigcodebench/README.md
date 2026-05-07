`bigcodebench` is wired as a thin one-step code-generation adapter over the shared MemRL brain.

Current scope:
- split: `instruct`
- default subset: `full`
- execution backend: official remote gradio evaluator
- optional diagnostic split files under `splits/`

Adapter boundary:
- loads benchmark tasks from the official annotation data
- exposes each task as one code-generation episode
- asks the shared brain for one Python solution
- submits that solution to the official evaluator backend
- converts `pass` / `fail` into MemRL reward and success

This adapter intentionally does not add benchmark-specific controller logic.

The split files in `splits/` are explicit diagnostic task selections used to test memory transfer, same-task replay, and domain-isolation behavior. They are not hidden defaults and are not required for the standard full-subset run. Report any result that uses a split file with the exact split filename, protocol, model, and evaluator route.
