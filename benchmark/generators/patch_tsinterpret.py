import os
import re
import TSInterpret

def patch_comte():
    base_dir = os.path.dirname(TSInterpret.__file__)
    target = os.path.join(base_dir, "InterpretabilityModels", "counterfactual", "COMTE", "Optimization.py")

    with open(target, "r") as f:
        code = f.read()

    # Change default distractors from 2 to 1
    code = re.sub(r'num_distractors\s*=\s*2', 'num_distractors=1', code)

    # Force threads=1
    code = re.sub(r'self\.threads\s*=\s*threads', 'self.threads = 1', code)

    with open(target, "w") as f:
        f.write(code)

    print(f"Patched file: {target}")

if __name__ == "__main__":
    patch_comte()
