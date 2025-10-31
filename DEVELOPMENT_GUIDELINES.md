# FED Watcher: Development Guidelines & Git Workflow

## 1. Guiding Principles

- **`main` is Production:** The `main` branch is for official, stable releases only. It must always be in a working state. Direct commits are forbidden.
- **`develop` is for Integration:** The `develop` branch is the single source of truth for active development work. All features and fixes are merged here.
- **Work in Isolation:** All work must be done on a dedicated `feature`, `bugfix`, or `hotfix` branch. Never commit directly to `main` or `develop`.
- **Review Everything:** All code must be peer-reviewed via a Pull Request before being merged into `develop`.

## 2. Branching Strategy: GitFlow

We adhere to the GitFlow branching model.

### Core Branches
- **`main`**: The production branch. Contains tagged, release-ready code.
- **`develop`**: The main development integration branch.

### Supporting Branches
- **`feature/*`**: For developing new features.
  - Branches from: `develop`
  - Merges into: `develop`
- **`bugfix/*`**: For fixing non-critical bugs found in development.
  - Branches from: `develop`
  - Merges into: `develop`
- **`release/*`**: For preparing a new milestone/release.
  - Branches from: `develop`
  - Merges into: `main` AND `develop`
- **`hotfix/*`**: For patching critical bugs found in `main` (production).
  - Branches from: `main`
  - Merges into: `main` AND `develop`

## 3. Standard Workflow (for Features & Bugfixes)

**Step 1: Start a New Task**
- Ensure your `develop` branch is up-to-date:
  git checkout develop
  git pull origin develop

- Start a new feature or bugfix branch using the `git flow` command. This will automatically create and check out the branch.
  # For a new feature
  git flow feature start [JIRA-ID]-[short-description]

  # For a non-critical bug
  git flow bugfix start [JIRA-ID]-[short-description]

**Step 2: Develop and Commit**
- Make small, logical commits to your branch.
- Use the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format: `type(scope): subject`.
  - **Examples:**
- `feat(data): Add script to download S&P 500 prices`
- `fix(nlp): Correct speaker attribution logic`

**Step 3: Finish Your Work (Open a Pull Request)**
- When your work is complete, use the `git flow` command to publish it.
  # For a feature
  git flow feature publish

  # For a bugfix
  git flow bugfix publish

- Go to GitHub and open a Pull Request.
- **Set the target branch to `develop`**.
- Assign at least one reviewer.

**Step 4: Review and Merge**
- Once the PR is approved, use GitHub's "Squash and Merge" button to merge it into `develop`.
- After merging, the branch can be deleted.

## 4. Code & Repository Rules

- **No Large Files:** Never commit data files (`.csv`, `.pkl`), environment files (`.env`), or secrets to Git. Use the `.gitignore` file to exclude them.
- **Clear Notebooks:** Before committing Jupyter Notebooks (`.ipynb`), always clear all cell outputs. We review code, not outputs.