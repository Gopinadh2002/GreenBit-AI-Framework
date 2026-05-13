# Data Folder

This folder contains sample datasets for testing GreenBit.

## sample_files/

Contains 12 sample text files for testing duplicate detection:

- **document1_draft.txt** - Original document draft
- **document1_draft_v2.txt** - Similar version (95% match)
- **document1_final.txt** - Final version (89% match)
- **machine_learning_basics.txt** - ML introduction
- **ml_advanced_topics.txt** - Advanced ML topics (87% match)
- **project_notes_meeting1.txt** - Meeting notes
- **project_notes_meeting1_backup.txt** - Backup (99% match)
- **api_documentation.txt** - API docs
- **api_documentation_v2.txt** - API docs v2 (92% match)
- **database_schema.txt** - Database design
- **requirements_analysis.txt** - Requirements
- **user_guide.txt** - User guide

## Expected Results

When analyzing sample_files/, GreenBit should detect:

- 3 document clusters (document versions)
- 1 duplicate pair (meeting notes)
- 2 similar doc pairs (API documentation)
- 1 similar pair (ML topics)

Total: ~4-6 duplicate/similar file groups
