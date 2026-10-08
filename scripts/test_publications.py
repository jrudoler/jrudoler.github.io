"""Regression checks for the public bibliography boundary and BibLaTeX mapping."""
import json
import unittest
from pathlib import Path

from generate_publications import convert, names


class PublicationExportTests(unittest.TestCase):
    def test_private_fields_and_manuscripts_do_not_publish(self):
        source = r'''@article{public,
 title = {A {Nested} Title}, author = {Rudoler, Joseph H.}, date = {2026},
 journaltitle = {Journal}, doi = {10.1234/paper},
 url = {https://journal.proxy.library.edu/paper},
 abstract = {private abstract}, file = {/Users/private.pdf},
 note = {private note}, keywords = {unread}}
@unpublished{private, title = {Secret manuscript}, author = {Rudoler, Joseph H.}, date = {2026}}
'''
        data = convert(source)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['title'], 'A Nested Title')
        self.assertEqual(data[0]['url'], 'https://doi.org/10.1234/paper')
        exported = json.dumps(data)
        for value in ('private abstract', '/Users/', 'private note', 'unread', 'Secret manuscript'):
            self.assertNotIn(value, exported)

    def test_accepted_work_and_arxiv_links(self):
        data = convert('''@inproceedings{paper, title={Title}, author={Rudoler, Joseph},
 date={2026}, booktitle={NeurIPS}, pubstate={inpress},
 url={http://arxiv.org/abs/2605.05436}, eprint={2605.05436}, eprinttype={arXiv}}
''')
        self.assertEqual(data[0]['type'], 'Paper')
        self.assertEqual(data[0]['status'], 'Accepted')
        self.assertEqual(data[0]['preprint'], '')
        self.assertEqual(data[0]['url'], 'https://arxiv.org/abs/2605.05436')

    def test_names_and_accents(self):
        self.assertEqual(names(r"Boix Adser\`a, Enric and {Research and Development Group}"),
                         ['Enric Boix Adserà', 'Research and Development Group'])

    def test_invalid_exports_fail(self):
        for source in ('', '@article{broken, title={unclosed}',
                       '@article{missing, title={Title}, date={2026}}',
                       '@article{a, title={T}, author={A}, date={2026}}\n'
                       '@article{a, title={T}, author={A}, date={2026}}'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                convert(source)

    def test_checked_in_snapshot(self):
        data = json.loads((Path(__file__).resolve().parents[1] / '_data/publications.json').read_text())
        self.assertEqual(len(data), 13)
        self.assertEqual(len({p['key'] for p in data}), 13)
        self.assertEqual({kind: sum(p['type'] == kind for p in data) for kind in
                         ('Paper', 'Preprint', 'Dataset', 'Presentation')},
                         {'Paper': 6, 'Preprint': 1, 'Dataset': 2, 'Presentation': 4})
        self.assertEqual(sum(p['status'] == 'Accepted' for p in data), 2)


if __name__ == '__main__':
    unittest.main()
