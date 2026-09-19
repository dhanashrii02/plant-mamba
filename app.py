import os
import sys

# Ensure core_engine modules are accessible
core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_engine")
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# Switch working directory to core_engine so relative assets resolve properly
os.chdir(core_dir)

from app import main

if __name__ == "__main__":
    main()
