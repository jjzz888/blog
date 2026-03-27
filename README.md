# blog

Jekyll blog for `https://jjzz888.github.io/blog`.

## Post a new blog

1. Create a file in `_posts/` named like:
   - `YYYY-MM-DD-my-post-title.md`
2. Add front matter and content:

```md
---
layout: post
title: "My Post Title"
date: 2026-03-27 10:00:00 +0800
categories: [notes]
---

Write your post here.
```

3. Commit and push to `gh-pages`:
   - `git checkout gh-pages`
   - `git add .`
   - `git commit -m "Add new blog post"`
   - `git push`

GitHub Pages will publish to `https://jjzz888.github.io/blog`.

## Optional local preview

If Ruby/Jekyll is installed:

- `bundle exec jekyll serve --baseurl "/blog"`
