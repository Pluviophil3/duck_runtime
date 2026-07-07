import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_runtime_constants(script_path):
    tree = ast.parse(script_path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (
                "JOINT_ORDER_16",
                "ACTION_ORDER_14",
                "ACTION_SCALE_BY_JOINT",
            ):
                constants[target.id] = ast.literal_eval(node.value)
    missing = {"JOINT_ORDER_16", "ACTION_ORDER_14", "ACTION_SCALE_BY_JOINT"} - set(constants)
    if missing:
        raise AssertionError(f"missing runtime constants: {sorted(missing)}")
    return constants


def xml_joint_order(xml_path):
    root = ET.parse(xml_path).getroot()
    return [j.attrib["name"] for j in root.iter("joint") if "name" in j.attrib]


def xml_action_order(xml_path):
    root = ET.parse(xml_path).getroot()
    names = [
        a.attrib.get("joint", a.attrib.get("name"))
        for a in root.iter("position")
        if a.attrib.get("joint", a.attrib.get("name"))
    ]
    return [name for name in names if "antenna" not in name]


def assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f"{name} mismatch:\nactual:   {actual}\nexpected: {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="mjlab_sway_t2_full_manifest.json",
        help="Path relative to this script directory unless absolute.",
    )
    parser.add_argument(
        "--xml",
        default="../../unitree_rl_mjlab/src/assets/robots/open_duck_mini_v2/xmls/open_duck_mini_v2.xml",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = script_dir / manifest_path
    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = (script_dir / xml_path).resolve()

    manifest = json.loads(manifest_path.read_text())
    runtime = load_runtime_constants(script_dir / "v2_rl_walk_mjlab.py")

    assert_equal("joint_order_16 constant", list(runtime["JOINT_ORDER_16"]), manifest["joint_order_16"])
    assert_equal("action_order_14 constant", list(runtime["ACTION_ORDER_14"]), manifest["action_order_14"])
    assert_equal(
        "action_order_14 derived from joint_order_16",
        [name for name in manifest["joint_order_16"] if "antenna" not in name],
        manifest["action_order_14"],
    )
    assert_equal("xml joint order", xml_joint_order(xml_path), manifest["joint_order_16"])
    assert_equal("xml action order without antennas", xml_action_order(xml_path), manifest["action_order_14"])

    runtime_scales = [runtime["ACTION_SCALE_BY_JOINT"][name] for name in runtime["ACTION_ORDER_14"]]
    assert_equal("action_scale_14", runtime_scales, manifest["action_scale_14"])

    obs_end = 0
    for term in manifest["observation_schema"]:
        if term["start"] != obs_end:
            raise AssertionError(f"observation term {term['name']} starts at {term['start']}, expected {obs_end}")
        if term["end"] - term["start"] != term["dim"]:
            raise AssertionError(f"observation term {term['name']} dim does not match start/end")
        obs_end = term["end"]
    if obs_end != manifest["policy"]["input_dim"]:
        raise AssertionError(f"observation ends at {obs_end}, expected {manifest['policy']['input_dim']}")

    policy_path = script_dir / manifest["policy"]["path"]
    motion_path = script_dir / manifest["motion"]["path"]
    assert_equal("policy sha256", sha256(policy_path), manifest["policy"]["sha256"])
    assert_equal("motion sha256", sha256(motion_path), manifest["motion"]["sha256"])

    motion = np.load(motion_path)
    if motion["joint_pos"].shape[1] != len(manifest["joint_order_16"]):
        raise AssertionError("motion joint_pos dim does not match joint_order_16")
    if motion["joint_vel"].shape[1] != len(manifest["joint_order_16"]):
        raise AssertionError("motion joint_vel dim does not match joint_order_16")
    if motion["body_quat_w"].shape[1] <= manifest["motion"]["anchor_body_index"]:
        raise AssertionError("motion body_quat_w does not contain anchor body index")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipped ONNX session shape check")
    else:
        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            raise AssertionError(f"expected one policy input, got {len(inputs)}")
        if inputs[0].name != manifest["policy"]["input_name"]:
            raise AssertionError(f"policy input name {inputs[0].name} != {manifest['policy']['input_name']}")
        test_obs = np.zeros((1, manifest["policy"]["input_dim"]), dtype=np.float32)
        action = session.run(None, {inputs[0].name: test_obs})[0]
        if action.shape[-1] != manifest["policy"]["output_dim"]:
            raise AssertionError(f"policy output shape {action.shape} does not end in output_dim")

    print("mjlab deploy verification passed")


if __name__ == "__main__":
    main()
