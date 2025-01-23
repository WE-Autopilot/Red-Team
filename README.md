# Git Best Practices

Below is how to commit to the repository. 
### Note: Please don't use GitHub desktop. If you have it, delete it please (please). The steps to use the terminal are much easier to debug for both me and yourself! 

---

### 1. Clone the Repository 📥
Start by cloning the repository to your local machine. Make a new folder entitled "WEAP" or something similar so you have all the club-related code in one location:
```bash
git clone https://github.com/WE-Autopilot/Red-MPC-controller
cd Red-MPC=Controller
```

### 2. Create a new feature branch  🌱
When commiting, it is important that you create a feature branch to avoid merge conflicts. Always work on a dedicated feature branch instead of main or dev. Use a descriptive name prefixed with feature/. Below is an example of how to make a feature branch:
```bash
#This new branch will track your changes related to a specific feature.
git checkout -b feature/your-feature-name  # e.g., feature/user-auth
```

### 3. Make and Commit Changes ✏️
Stage and commit your changes with clear, concise messages:
```bash
git add .  # Stage all changes (or specify individual files)
git commit -m "Describe your changes briefly but meaningfully"
```
Run this command after commiting to sync your branch with the latest version of dev:
```bash
git pull origin dev
```

### 4. Push Changes and Create a Pull Request (PR) 🚀
Push your feature branch to the remote repository:
```bash
git push origin feature/your-feature-name
```
Navigate to GitHub and create a Pull Request (PR) from your feature branch to the dev branch.

### 5. Testing! 🧪
I need to add tests later......... Don't worry about it for now

### 6. Merge the Pull Request 🔀
After your PR is approved merge it into dev
Pull the latest dev branch locally:

```bash
git checkout dev
git pull origin dev
```

### 7. Delete Merged Branches 🧹
Delete the remote feature branch via GitHub after merging.

```bash
git branch -d feature/your-feature-name
```
Clean up outdated remote references:
```bash
git fetch --prune
```
