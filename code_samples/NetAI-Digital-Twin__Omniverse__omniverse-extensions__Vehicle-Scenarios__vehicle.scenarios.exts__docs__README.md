## Scenario Description
https://www.notion.so/Scenario-2ac03d54bf4a804b966ac93ee966e702?source=copy_link

![vehicle scenarios](https://github.com/user-attachments/assets/21f55419-544c-4684-bfc8-2d4ae45aabce)


# Vehicle Scenarios Extension

## Installation (Clone This Extension)

1. **Create a download folder**
   ```bash
   mkdir omniverse-extensions
   cd omniverse-extensions
   ```

2. **Initialize local repository**
   ```bash
   git init
   git remote add -f origin https://github.com/KappyDays/omniverse-extensions.git
   ```

3. **Enable sparse-checkout**
   ```bash
   git sparse-checkout init --cone
   ```

4. **Checkout Vehicle-Scenarios folder only**
   ```bash
   git sparse-checkout set Vehicle-Scenarios
   ```

5. **Download the selected folder**
   ```bash
   git pull origin main
   ```

## Importing the Vehicle Scenarios Extension

1. Add the **parent directory** of the cloned repository to the **Extension Search Paths** in the Extensions window.
   - Navigate to: **Extensions > Gear Icon (Settings) > Extension Search Paths**
2. Verify that **"SIX VEHICLE SCENARIOS"** appears under **Extensions > THIRD PARTY > User**.

## Setting Up Vehicle Scenario Simulations

1. Open **Extensions**, search for `actor`, and enable **ACTOR SDG UI**.
2. In the **ACTOR SDG UI**, configure the **Config File Path**:
   ```
   omniverse://[YOUR_IP]/Projects/Dream-AI_Plus_Twin/Workspace_Personal/kkr/course_work/AECO_CityDemoPack_NVD@10011/Demos/AEC/TowerDemo/CityDemopack/Setup_files/kkr_config.yaml
   ```
3. Click **"Set UP Simulation"** in the ACTOR SDG UI.
   - If a warning dialog appears, click **Yes**.
4. Activate **Six Vehicle Scenarios** from the **Menu Bar**.

> **Note:** If you cannot access the NetAI Nucleus server, you will need to prepare the simulation scene assets and character command file manually.

## Running Vehicle Scenario Tests

1. **Select a scenario (1-5)**.
   - Selecting "First-person" switches the view to the camera mounted on the front of the vehicle.
2. **Start the scenario**: Press **Play** or hit the `SPACE` bar.
> To re-run a scenario, **Stop** the simulation, select a other scenario, and press **Play** again.
