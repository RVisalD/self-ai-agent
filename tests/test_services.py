import json
import unittest
from unittest.mock import Mock, patch

from services import local_clock, search_web


class ServiceTests(unittest.TestCase):
    def test_clock_validation(self):
        self.assertIn('Unknown timezone', local_clock('Bad/TimeZone'))
        self.assertNotIn('Unknown timezone', local_clock('Asia/Phnom_Penh'))

    def test_empty_query(self):
        self.assertIn('provide a search query', search_web('  '))

    @patch.dict('os.environ', {'TAVILY_API_KEY': ''})
    def test_missing_key(self):
        self.assertIn('not configured', search_web('weather Phnom Penh'))

    @patch.dict('os.environ', {'TAVILY_API_KEY': 'test-key'})
    @patch('tavily.TavilyClient')
    def test_search_preserves_sources_and_dates(self, client):
        client.return_value.search.return_value = {'results': [
            {'title': 'Forecast', 'url': 'https://example.com/weather',
             'content': 'Rain expected', 'published_date': '2026-10-24'},
            {'title': 'Missing URL'},
        ]}
        output = json.loads(search_web('  Phnom Penh weather October 24 2026  '))
        self.assertEqual(len(output['results']), 1)
        self.assertEqual(output['results'][0]['url'], 'https://example.com/weather')
        self.assertEqual(output['results'][0]['published_date'], '2026-10-24')
        self.assertEqual(client.return_value.search.call_args.kwargs['topic'], 'general')
        self.assertEqual(output['query'], 'Phnom Penh weather October 24 2026')

    @patch.dict('os.environ', {'TAVILY_API_KEY': 'test-key'})
    @patch('tavily.TavilyClient')
    def test_no_results(self, client):
        client.return_value.search.return_value = {'results': []}
        self.assertIn('No relevant web results', search_web('latest news'))

    @patch.dict('os.environ', {'TAVILY_API_KEY': 'test-key'})
    @patch('tavily.TavilyClient')
    def test_provider_failure(self, client):
        client.return_value.search.side_effect = RuntimeError('private provider details')
        output = search_web('latest news')
        self.assertIn('Live search was unavailable', output)
        self.assertNotIn('private provider details', output)


if __name__ == '__main__':
    unittest.main()
