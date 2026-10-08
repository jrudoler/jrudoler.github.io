# Publication list

Publication metadata is maintained in Zotero's `my-pubs` collection and exported
by Better BibLaTeX into the CV repository's `my-pubs.bib`. Do not maintain a second
bibliography in this repository.

The CV publishing workflow runs `scripts/generate_publications.py` from website
`master`, producing a public `publications.json` artifact alongside the CV PDF.
The website deployment loads that artifact into `_data/publications.json` **before**
building Jekyll. This pairs the online CV and publication metadata, and prevents
later website commits from replacing newer published metadata with an old snapshot.
An invalid export fails publication before either artifact is pushed.

The checked-in JSON is a local preview/bootstrap snapshot. Until the updated CV
publisher first runs, deployments use this snapshot. Deploy website changes first,
then the updated CV workflow; both repositories currently have local changes only.
Changing the exporter requires a subsequent CV workflow run to refresh the artifact.

## Local preview

From the website repository:

```sh
python3 -m venv /private/tmp/website-publications-venv
/private/tmp/website-publications-venv/bin/pip install -r scripts/publications-requirements.txt
/private/tmp/website-publications-venv/bin/python scripts/generate_publications.py \
  ../RudolerCV/my-pubs.bib _data/publications.json
bundle exec jekyll serve --host 127.0.0.1 --port 4000 \
  --destination /private/tmp/rudoler-publications-site
```

Open http://localhost:4000/publications/. Stop any existing server on port 4000
before running Jekyll serve. The separate destination keeps preview builds out of
tracked `_site` files.

## Website-specific details

Edit `_data/publication_details.yml`, keyed by the existing Zotero citation keys.
Supported fields:

```yaml
rudolerEstimatingImplicitRegularization2026:
  selected: true
  topics: [Deep Learning Theory]
  summary: Optional short explanation of the contribution.
  links:
    - label: Code
      url: https://github.com/your-project
```

New publications appear automatically even without an editorial entry, but will
have no topic tags and will not appear in Selected until configured. Topics are
curated here so personal Zotero tags such as `unread` never become public filters.
Keep citation keys stable; metadata here does not override bibliographic fields.

The exporter includes papers, preprints, presentations, and datasets. It deliberately
excludes `@unpublished` manuscripts in preparation. Attachment paths, abstracts,
notes, and library tags are not exported, including in the expandable citations.
Institutional proxy URLs fall back to public DOI links. Presentation collaborators
are shown separately rather than silently treated as authors.

All entries and citations are server-rendered and readable without JavaScript.
JavaScript enables Selected/All, a single topic filter, publication type,
and newest/oldest/title sorting. These filters combine, and an empty result offers
a Clear filters button.

The layout is inspired by Martin Saveski's Jekyll publication list,
credited on the page. The implementation is original
and uses the existing site's theme.
