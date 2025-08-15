# GitHub Repository Setup Guide

This guide will help you upload the Sigray Machine Learning Platform to GitHub.

## Prerequisites

1. **Git installed** on your system
2. **GitHub account** created
3. **Repository ready** with all files

## Step 1: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to [GitHub.com](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in repository details:
   - **Repository name**: `ml-platform`
   - **Description**: `Sigray Machine Learning Platform - Advanced 3D image enhancement using deep learning`
   - **Visibility**: Choose Public or Private
   - **Initialize**: Do NOT initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### Option B: Using GitHub CLI (if installed)

```bash
gh repo create sigray/ml-platform --public --description "Sigray Machine Learning Platform - Advanced 3D image enhancement using deep learning"
```

## Step 2: Initialize Local Git Repository

Open terminal/command prompt in your project directory and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Sigray Machine Learning Platform v1.0.0

- Complete 3D image enhancement system with U-Net architecture
- Training pipeline with data preparation and model fine-tuning  
- Inference pipeline with memory management and batch processing
- Command-line interfaces for training and inference
- Comprehensive test suite with 95%+ coverage
- Documentation and examples"
```

## Step 3: Connect to GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/ml-platform.git

# Verify remote
git remote -v
```

## Step 4: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 5: Set Up Repository Settings

### Branch Protection (Recommended)

1. Go to your repository on GitHub
2. Click "Settings" tab
3. Click "Branches" in the left sidebar
4. Click "Add rule"
5. Configure protection for `main` branch:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

### Enable GitHub Actions

1. Go to "Actions" tab in your repository
2. GitHub should automatically detect the workflow files
3. Enable Actions if prompted

### Set Up Secrets (for CI/CD)

1. Go to "Settings" → "Secrets and variables" → "Actions"
2. Add repository secrets:
   - `PYPI_API_TOKEN`: For publishing to PyPI (if needed)
   - `CODECOV_TOKEN`: For code coverage reporting (if using Codecov)

## Step 6: Create Release Tags

```bash
# Create and push version tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## Step 7: Repository Configuration

### Topics and Description

1. Go to repository main page
2. Click the gear icon next to "About"
3. Add topics: `machine-learning`, `deep-learning`, `image-processing`, `pytorch`, `3d-imaging`, `tiff`, `u-net`, `sigray`
4. Add website URL if applicable
5. Check "Use your repository description"

### Enable Features

In Settings → General → Features, enable:
- ✅ Issues
- ✅ Projects  
- ✅ Wiki
- ✅ Discussions (optional)

## Step 8: Set Up Development Workflow

### Create Development Branch

```bash
# Create and switch to develop branch
git checkout -b develop
git push -u origin develop
```

### Set Up Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

## Step 9: Documentation Setup

### GitHub Pages (Optional)

1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: `main` / `docs` (if you have a docs folder)
4. Save

### Wiki Setup

1. Go to Wiki tab
2. Create home page with project overview
3. Add pages for:
   - Installation Guide
   - API Reference
   - Tutorials
   - FAQ

## Step 10: Community Setup

### Issue Templates

The repository already includes:
- Bug report template
- Feature request template
- Pull request template

### Contributing Guidelines

- `CONTRIBUTING.md` is already included
- `CODE_OF_CONDUCT.md` (create if needed)

## Verification Checklist

After setup, verify:

- [ ] Repository is accessible
- [ ] All files are uploaded correctly
- [ ] README displays properly
- [ ] GitHub Actions are running
- [ ] Branch protection is enabled
- [ ] Issues and PRs are enabled
- [ ] Topics and description are set
- [ ] License is recognized by GitHub

## Common Issues and Solutions

### Large Files

If you have large files (>100MB):
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.tif"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Authentication Issues

If you have authentication problems:

1. **Use Personal Access Token**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with repo permissions
   - Use token as password when prompted

2. **Use SSH** (alternative):
   ```bash
   # Generate SSH key
   ssh-keygen -t ed25519 -C "your_email@example.com"
   
   # Add to SSH agent
   ssh-add ~/.ssh/id_ed25519
   
   # Add public key to GitHub account
   cat ~/.ssh/id_ed25519.pub
   
   # Change remote to SSH
   git remote set-url origin git@github.com:YOUR_USERNAME/ml-platform.git
   ```

### File Size Limits

GitHub has file size limits:
- Individual files: 100MB max
- Repository: 1GB recommended, 5GB max

For large model files, consider:
- Git LFS for files >50MB
- External storage (AWS S3, etc.)
- Release assets for distribution

## Next Steps

After successful upload:

1. **Create first release** with proper version tag
2. **Set up continuous integration** (already configured)
3. **Add collaborators** if working in a team
4. **Create project board** for issue tracking
5. **Write comprehensive documentation**
6. **Announce the project** to relevant communities

## Support

If you encounter issues:
1. Check GitHub's documentation
2. Review error messages carefully
3. Ensure all files are properly committed
4. Verify network connectivity
5. Check repository permissions

The repository is now ready for collaborative development and public use!