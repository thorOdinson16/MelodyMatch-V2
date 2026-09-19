from pathlib import Path
from datetime import datetime
import subprocess
import sys


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ============================================================
# Experiments
# ============================================================

EXPERIMENTS = [
    (
        "CNN + BiLSTM",
        "train_cnn_bilstm.py",
    ),
    (
        "CNN + Transformer",
        "train_cnn_transformer.py",
    ),
    (
        "CNN + Transformer + Metadata",
        "train_transformer_metadata.py",
    ),
]


# ============================================================
# Run one experiment
# ============================================================

def run_experiment(name, script_name):

    script_path = SCRIPTS_DIR / script_name

    print("\n")
    print("=" * 80)
    print(f"STARTING: {name}")
    print(f"Script:   {script_name}")
    print(f"Started:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Make sure the script exists
    # --------------------------------------------------------

    if not script_path.exists():

        print(
            f"ERROR: Script does not exist:\n"
            f"{script_path}"
        )

        return {
            "name": name,
            "script": script_name,
            "status": "FAILED",
            "exit_code": -1,
            "duration": None,
        }

    start_time = datetime.now()

    try:

        # ----------------------------------------------------
        # Run the training script
        #
        # This blocks until the experiment finishes.
        # Therefore only ONE model uses the GPU at a time.
        # ----------------------------------------------------

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )

        exit_code = result.returncode

    except KeyboardInterrupt:

        print("\n")
        print("=" * 80)
        print(
            f"INTERRUPTED BY USER: {name}"
        )
        print("=" * 80)

        return {
            "name": name,
            "script": script_name,
            "status": "INTERRUPTED",
            "exit_code": -2,
            "duration": datetime.now() - start_time,
        }

    except Exception as error:

        print("\n")
        print("=" * 80)
        print(
            f"ERROR WHILE RUNNING: {name}"
        )
        print(error)
        print("=" * 80)

        return {
            "name": name,
            "script": script_name,
            "status": "FAILED",
            "exit_code": -1,
            "duration": datetime.now() - start_time,
        }

    end_time = datetime.now()
    duration = end_time - start_time

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if exit_code == 0:

        status = "SUCCESS"

        print("\n")
        print("=" * 80)
        print(f"COMPLETED: {name}")
        print(f"Duration:  {duration}")
        print(f"Exit code: {exit_code}")
        print("=" * 80)

    else:

        status = "FAILED"

        print("\n")
        print("=" * 80)
        print(f"FAILED: {name}")
        print(f"Duration:  {duration}")
        print(f"Exit code: {exit_code}")
        print("=" * 80)

    return {
        "name": name,
        "script": script_name,
        "status": status,
        "exit_code": exit_code,
        "duration": duration,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("MELODYMATCH V2")
    print("SEQUENTIAL AUDIO EXPERIMENT RUNNER")
    print("=" * 80)

    print()
    print(f"Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Experiments:")
    
    for index, (name, script) in enumerate(
        EXPERIMENTS,
        start=1,
    ):
        print(
            f"  {index}. {name}"
        )
        print(
            f"     {script}"
        )

    print()
    print(
        "Each experiment will run sequentially."
    )
    print(
        "The next experiment starts only after "
        "the previous one finishes."
    )

    print()
    print("=" * 80)

    overall_start = datetime.now()

    results = []

    # ========================================================
    # Sequential execution
    # ========================================================

    for name, script_name in EXPERIMENTS:

        result = run_experiment(
            name,
            script_name,
        )

        results.append(result)

        # ----------------------------------------------------
        # Continue after failure
        # ----------------------------------------------------

        if result["status"] == "SUCCESS":

            print(
                f"\n✓ {name} finished successfully."
            )

        elif result["status"] == "FAILED":

            print(
                f"\n✗ {name} failed."
            )

            print(
                "Continuing with the next experiment..."
            )

        elif result["status"] == "INTERRUPTED":

            print(
                "\nExperiment runner was "
                "interrupted by the user."
            )

            print(
                "Stopping the remaining experiments."
            )

            break

    overall_end = datetime.now()
    overall_duration = (
        overall_end - overall_start
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n\n")
    print("=" * 80)
    print("EXPERIMENT RUN COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Total runtime: {overall_duration}"
    )

    print()
    print("Results:")
    print("-" * 80)

    for result in results:

        print(
            f"{result['status']:12s} | "
            f"{result['name']}"
        )

        print(
            f"{'':12s} | "
            f"Exit code: {result['exit_code']}"
        )

        if result["duration"] is not None:

            print(
                f"{'':12s} | "
                f"Duration: {result['duration']}"
            )

    print("-" * 80)

    successful = sum(
        result["status"] == "SUCCESS"
        for result in results
    )

    failed = sum(
        result["status"] == "FAILED"
        for result in results
    )

    interrupted = sum(
        result["status"] == "INTERRUPTED"
        for result in results
    )

    print(
        f"Successful:  {successful}"
    )

    print(
        f"Failed:      {failed}"
    )

    print(
        f"Interrupted: {interrupted}"
    )

    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()