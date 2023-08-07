from . import m2023_07_30 as current_run

if __name__ == "__main__":
    print(f"Starting run {current_run.__name__}")

    debug_info = current_run.main()
