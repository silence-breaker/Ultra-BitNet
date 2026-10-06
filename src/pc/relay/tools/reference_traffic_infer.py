import json
import os
import torch

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("BITNET_MODEL", "./src/models/official_model")
tokenizer = AutoTokenizer.from_pretrained(MODEL, fix_mistral_regex=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
)

system = (
    "Return one JSON object only. Exact keys: camera_id,timestamp,traffic_overview,"
    "abnormal_events,traffic_flow. camera_id must always be camera_0. timestamp, "
    "direction and vehicle_count must be copied exactly from INPUT. Never invent values. "
    "If has_event=false, abnormal_events must be No abnormal events detected."
)
example_input = {
    "timestamp": "2026-01-01 10:00:00",
    "event": {"has_event": False},
    "traffic_flow": {"direction": "east", "vehicle_count": 3},
}
example_output = {
    "camera_id": "camera_0",
    "timestamp": "2026-01-01 10:00:00",
    "traffic_overview": "The intersection has 3 eastbound vehicles; traffic is free-flowing.",
    "abnormal_events": "No abnormal events detected.",
    "traffic_flow": {"direction": "east", "vehicle_count": 3},
}
report = {
    "schema_version": "2.0",
    "timestamp": "2026-03-13 03:41:31",
    "event": {
        "has_event": False,
        "event_type": "none",
        "participants": [],
        "participant_classes": [],
        "location": [],
        "desc": "",
    },
    "traffic_flow": {"direction": "north", "vehicle_count": 0},
}
user = "EXAMPLE INPUT: " + json.dumps(example_input, separators=(",", ":"))
user += " EXAMPLE OUTPUT: " + json.dumps(example_output, separators=(",", ":"))
user += " INPUT: " + json.dumps(report, separators=(",", ":")) + " OUTPUT:"
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print("PROMPT=" + prompt)
print("OUTPUT=" + tokenizer.decode(outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
