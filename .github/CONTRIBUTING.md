## Branching Naming Conventions
We follow the Conventional Commits naming pattern. All branches should be named in `kebab-case` using the following format:

`type/short-description` or `type/issue-number-description`

Allowed prefixes:
* feat/ - New features or gameplay mechanics
* fix/ - Bug fixes
* docs/ - Documentation changes
* refactor/ - Code refactoring without logic changes
* perf/ - Performance optimizations
* test/ - Adding or updating tests

Branch names are validated _automatically_ against this pattern:

```
^(feat|fix|docs|refactor|perf|test)\/[a-z0-9-]+$
```

### General Rules for Branch Names
* Use lowercase letters only.
* Use (`-`) to separate words (kebab-case). Do not use spaces or underscores.
* Keep names concise but descriptive (2–4 words max).
* Always branch off the `master` branch.