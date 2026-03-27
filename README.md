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
categories: [notes/test/...]
---

Write your post here.
```

3. Commit and push to the `gh-pages` branch

```bash
git checkout gh-pages
git add _posts/YYYY-MM-DD-your-post.md _config.yml  # if you changed config
git commit -m "Add new blog post: <title>"  # or "Update permalink pattern"
git push origin gh-pages
```

Because `baseurl: "/blog"` is already set, the final published URL for a post with slug `my-second-post` will be:

```
https://jjzz888.github.io/blog/my-second-post.html

```

Notes and best practices:
- Keep filenames in `_posts/` in the Jekyll format `YYYY-MM-DD-title.md`. The `:title` used in the permalink is the title-slug derived from the filename (the part after the date).
- Make titles (and filenames) unique and slug-friendly (lowercase, hyphens instead of spaces) to avoid permalink conflicts.
- If you need a custom URL for a specific post, add a `permalink:` entry to that post's front matter, for example:

```md
---
layout: post
title: "My Post Title"
date: 2026-03-27 10:00:00 +0800
categories: [notes/test/...]
permalink: /notes/my-post-custom.html
---
```

- If you prefer to include a category path (for example `/notes/<title>.html`), you can instead set in `_config.yml`:

```yaml
permalink: /:categories/:title.html
```

This will publish posts with `categories: [notes]` at `/notes/<title>.html` (and, with `baseurl`, at `https://jjzz888.github.io/blog/notes/<title>.html`).
