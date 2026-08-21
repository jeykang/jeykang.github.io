# FPS Measurement Extension

![isaacsim framerate measurement](https://github.com/user-attachments/assets/e0a5a34d-3aef-465c-a152-66c5f29c4005)

This extension records the camera movement path of the Perspective view controlled by the user and allows for playback. It also measures the FPS (Frames Per Second) while replaying the recorded path.

## Installation

You can install this extension in two ways: cloning the entire repository or cloning only the specific extension folder.

### Method 1: Clone All Extensions
To download the entire `omniverse-extensions` repository:

```bash
git clone https://github.com/KappyDays/omniverse-extensions.git
```

### Method 2: Clone This Extension Only (Sparse Checkout)

If you only want to download the `isaacsim.framerate.measurement` folder (Recommended if you want to save space, same logic as Folder A):

1.  **Create a download folder**

    ```bash
    mkdir omniverse-extensions
    cd omniverse-extensions
    ```

2.  **Initialize local repository**

    ```bash
    git init
    git remote add -f origin [https://github.com/KappyDays/omniverse-extensions.git](https://github.com/KappyDays/omniverse-extensions.git)
    ```

3.  **Enable sparse-checkout**

    ```bash
    git sparse-checkout init --cone
    ```

4.  **Checkout specific folder only**

    ```bash
    git sparse-checkout set isaacsim.framerate.measurement
    ```

5.  **Download the selected folder**

    ```bash
    git pull origin main
    ```

## Importing the Extension

1.  Add the **parent directory** of the cloned repository to the **Extension Search Paths** in the Extensions window.
      - Navigate to: **Extensions \> Gear Icon (Settings) \> Extension Search Paths**
2.  Verify that the extension appears under **Extensions \> THIRD PARTY \> User**.

## Usage

1.  **Activate the Extension Window**

      - Click **KKR-Tools** on the right side of the **Window** tab.
      - Select **FPS Measure** to open the extension window.

2.  **Record Camera Movement**

      - Click **Start** to begin recording the camera movement path in the Perspective view.
      - Click **Stop** to finish recording.

3.  **Replay Recording**

      - Click **Play Recording** to replay the recorded camera path and measure the FPS.

4.  **Save & Load Path (JSON)**

      - **Save:** Select a folder to save the recorded path and click the **Save JSON** button. The path will be saved in JSON format.
      - **Load:** Select a previously saved path and click **Load JSON** to retrieve the camera movement path.
