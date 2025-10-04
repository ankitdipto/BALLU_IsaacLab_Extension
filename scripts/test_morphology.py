import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
ext_dir = os.path.join(project_dir, "source", "ballu_isaac_extension", "ballu_isaac_extension")
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

try:
    import morphology  # noqa: F401
    from morphology import BalluMorphology, BalluRobotGenerator
except Exception as exc:
    print("[ERROR] Failed to import morphology:", exc)
    print("[HINT] Ensure PYTHONPATH includes:", ext_dir)
    raise Exception(f"Failed to import morphology: {exc}")


def main():

    # Create default morphology and validate
    morph = BalluMorphology.default()
    ok, errs = morph.validate()
    print("Morphology ID:", morph.morphology_id)
    print("Validation:", ok)
    if not ok:
        print("Validation errors:")
        for e in errs:
            print(" -", e)

    # Instantiate generator; override URDF output dir to local path. Do NOT generate USD (needs Isaac).
    G = BalluRobotGenerator(morph)
    urdf_path = G.generate_urdf()
    print("URDF written:", urdf_path)
    ret_code = G.generate_usd(urdf_path)
    print("USD conversion return code:", ret_code)
    # Echo a few derived properties
    props = morph.get_derived_properties()
    print("Total leg length:", round(props["total_leg_length"], 6))
    print("Femur ratio:", round(props["femur_to_limb_ratio"], 6))
    print("Total mass:", round(props["total_mass"], 6))


if __name__ == "__main__":
    main()


