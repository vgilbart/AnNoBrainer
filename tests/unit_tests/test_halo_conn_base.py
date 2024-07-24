import sys
import unittest
import responses
sys.path.append(".")
import backend_helpers.halo_api_handler.halo_api_conn_base as conn_base


class TestCase(unittest.TestCase):

    @responses.activate
    def test_receiving_token(self):
        halo_api_base_instance = conn_base.ConnectorHaloOidc("http://localhost:4030/token",
                                                         "http://localhost:4030/graphql",
                                                         b'dGVzdF91c2VybmFtZQ==', b'dGVzdF9wYXNzd29yZA==')

        responses.add(**{
            'method': responses.POST,
            'url': 'http://localhost:4030/token',
            'body': '{"authn": "sample_token"}',
            'status': 200
        })
        halo_api_base_instance.request_token()
        self.assertEqual(halo_api_base_instance.username, b'dGVzdF91c2VybmFtZQ==')
        self.assertEqual(halo_api_base_instance.password, b'dGVzdF9wYXNzd29yZA==')
        self.assertEqual(halo_api_base_instance.token, 'sample_token')

    @responses.activate
    def test_receiving_graphql_response(self):
        halo_api_base_instance = conn_base.ConnectorHaloOidc("http://localhost:4030/token",
                                                         "http://localhost:4030/graphql",
                                                         b'dGVzdF91c2VybmFtZQ==', b'dGVzdF9wYXNzd29yZA==')
        halo_api_base_instance.token = "sample_token"

        responses.add(**{
            'method': responses.POST,
            'url': 'http://localhost:4030/graphql',
            'body': '{"img_id": "123456789"}',
            'status': 200
        })
        halo_api_base_instance.request_data("graphql_query", "/path/to/image.tif")
        self.assertEqual(halo_api_base_instance.query_output, {'img_id': '123456789'})


if __name__ == "__main__":
    unittest.main()