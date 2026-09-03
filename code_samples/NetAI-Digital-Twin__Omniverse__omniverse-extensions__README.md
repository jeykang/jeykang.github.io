# Omniverse Extensions

## How to Import These Extensions

Navigate to: **Isaac Sim Extensions → Settings → Extension Search Paths**

### Method 1: Direct Import via Git
Import extensions by simply providing a Git URL. Omniverse will automatically clone the repository and register the extension without requiring manual download.

**Prerequisites:** Git must be installed on your system.

**Example: Importing Vehicle-Scenarios Extension**
```
git://github.com/KappyDays/NetAI-Digital-Twin.git?branch=main&dir=Omniverse/omniverse-extensions/Vehicle-Scenarios
```

### Method 2: Local Import
Download the extension repository (folder) and register its local path in the Extension Search Paths.

**Example: Importing Vehicle-Scenarios Extension**
1. Download the `Vehicle-Scenarios` folder
2. Add the local path to Extension Search Paths
   ```
   C:\workspace\NetAI-Digital-Twin\Omniverse\omniverse-extensions\Vehicle-Scenarios
   ```
