import os
import sys
import runpy

# Ensure core_engine modules are accessible
core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_engine")
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# Switch working directory to core_engine so relative assets resolve properly
os.chdir(core_dir)

# Execute core_engine/app.py cleanly in the global namespace
target_app = os.path.join(core_dir, "app.py")
runpy.run_path(target_app, run_name="__main__")
