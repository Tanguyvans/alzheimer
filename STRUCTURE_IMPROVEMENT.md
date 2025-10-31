# Project Structure Improvement Analysis

## Current Issues

### 1. **Too Many Root-Level .md Files** (Messy!)
```
Root directory has 5 markdown files:
- README.md (main)
- CLAUDE.md (AI instructions)
- ALZHEIMER_RESEARCH_GROUPS.md (ADNI group definitions)
- PROJECT_SUMMARY.md (old daily summary from Oct 24)
- EXPERIMENTS_GUIDE.md (experiments overview)
```

### 2. **Duplicate/Outdated Documentation**
- `PROJECT_SUMMARY.md` - Old summary from Oct 24, now outdated
- `EXPERIMENTS_GUIDE.md` - Overlaps with `experiments/README.md`
- `ALZHEIMER_RESEARCH_GROUPS.md` - Could go in `docs/`

### 3. **Empty/Sparse Folders**
- `docs/preprocessing/` - Empty
- `docs/mri/` - Empty
- Both could be removed or merged into main folders

---

## Proposed Clean Structure

### Root Directory (Keep ONLY Essential Files)
```
alzheimer/
├── README.md                    # Main project README (keep)
├── CLAUDE.md                    # AI instructions (keep)
├── requirements.txt             # Dependencies (keep)
│
├── preprocessing/               # All preprocessing code + docs
├── experiments/                 # All experiments + guides
├── tabular/                     # Tabular models
├── models/                      # Trained models
├── utils/                       # Utilities
├── data/                        # Data files
└── docs/                        # Consolidated documentation
```

### Move Documentation

**docs/** (consolidated):
```
docs/
├── README.md                    # Index of all docs
├── datasets/                    # Dataset docs (keep existing)
│   ├── ADNI_GUIDE.md
│   └── ...
├── research/                    # Research reference docs (NEW)
│   └── ADNI_RESEARCH_GROUPS.md  # Move from root
└── tabular/                     # Keep existing
    └── METRICS_GUIDE.md
```

---

## Recommended Actions

### ✅ Action 1: Remove Outdated/Duplicate Files

**Remove** (3 files):
```bash
rm PROJECT_SUMMARY.md              # Outdated (Oct 24)
rm EXPERIMENTS_GUIDE.md            # Duplicate of experiments/README.md
rm ALZHEIMER_RESEARCH_GROUPS.md    # Move to docs/research/
```

### ✅ Action 2: Clean Empty Folders

**Remove** (2 empty folders):
```bash
rm -rf docs/preprocessing/  # Empty
rm -rf docs/mri/            # Empty
```

### ✅ Action 3: Consolidate Documentation

**Move ADNI groups to docs**:
```bash
mkdir -p docs/research
# (keep ALZHEIMER_RESEARCH_GROUPS.md but move to docs/research/)
```

---

## Final Clean Structure

### Root Directory (After Cleanup)
```
alzheimer/
├── README.md                    ✅ Main entry point
├── CLAUDE.md                    ✅ AI instructions
├── requirements.txt             ✅ Dependencies
│
├── preprocessing/               ✅ All preprocessing (+ README.md)
│   ├── README.md
│   ├── pipeline_1_synthstrip/
│   ├── pipeline_2_nppy/
│   └── ...
│
├── experiments/                 ✅ All experiments (+ README.md)
│   ├── README.md
│   ├── NPPY_SOLUTION_SUMMARY.md
│   ├── cn_vs_ad_3dhcct/
│   └── ...
│
├── tabular/                     ✅ Tabular models (+ METRICS_GUIDE.md)
├── models/                      ✅ Trained models (+ README.md)
├── models_2d/                   ✅ 2D models (+ README.md)
├── model_3d/                    ✅ 3D models (+ README.md + TRAINING_GUIDE.md)
├── utils/                       ✅ Utilities (visualize.py, gen_tabular.py)
├── data/                        ✅ Data files
│
└── docs/                        ✅ Consolidated docs
    ├── README.md                    (index)
    ├── datasets/                    (ADNI guides)
    ├── research/                    (reference docs)
    │   └── ADNI_RESEARCH_GROUPS.md  (moved from root)
    └── tabular/
        └── METRICS_GUIDE.md
```

---

## Benefits of This Structure

1. **Cleaner Root** - Only 3 files instead of 6
2. **No Duplication** - One place for each type of doc
3. **Logical Grouping** - Docs organized by topic in `docs/`
4. **Easy Navigation** - Clear hierarchy
5. **Maintainable** - Each folder has its own README

---

## Implementation Commands

```bash
# 1. Remove outdated files
rm PROJECT_SUMMARY.md EXPERIMENTS_GUIDE.md

# 2. Move research docs
mkdir -p docs/research
mv ALZHEIMER_RESEARCH_GROUPS.md docs/research/

# 3. Remove empty folders
rm -rf docs/preprocessing docs/mri

# 4. Done! Clean structure achieved.
```

---

## Summary

**Remove**: 2 outdated .md files from root
**Move**: 1 file to docs/research/
**Delete**: 2 empty folders in docs/

**Result**: Clean, logical, maintainable structure with proper documentation hierarchy.
