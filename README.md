
### Project setup (Ubuntu):
1. Install `uv` (if not already installed): `pip install uv`
2. Create a virtual environment: `uv venv`
3. Activate it: `source .venv/bin/activate`
4. Install dependencies and the project in editable mode: `uv pip install -e .`
5. Check if it is working: `python -c "import dqn.dqn_config"`
  - If nothing is printed and no error appears, it means the project is installed correctly in editable mode, and imports are working properly.


### Project setup (Windows)
1. Install `uv` (if not already installed): `pip install uv`
2. Create a virtual environment: `uv venv`
3. Activate it: `.venv\Scripts\Activate`
4. Install dependencies and the project in editable mode: `uv pip install -e .`
5. Check if it is working: `python -c "import dqn.dqn_config"`
  - If nothing is printed and no error appears, it means the project is installed correctly in editable mode, and imports are working properly.
