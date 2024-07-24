import sys
import unittest
import responses

from collections import namedtuple

sys.path.append(".")
import backend_helpers.halo_api_handler.graphql_connector as gql_conn


class TestCase(unittest.TestCase):

    @responses.activate
    def test_receiving_graphql_response(self):
        responses.add(**{
            'method': responses.POST,
            'url': 'http://localhost:4030/graphql/',
            'body': '{"data": {"imageById": {"id": "SW1hZ2U6MzY5OQ==", "location": "output_file"}}}',
            'status': 200
        })
        responses.add(**{
            'method': responses.POST,
            'url': 'http://localhost:4030/token/',
            'body': '{"authn": "sample_token"}',
            'status': 200
        })

        halo_params = namedtuple('api_conf', ["url", "auth_path", "login_path", "api_path", "verify", "client_id",
                                                   "client_secret", "type", "connected", "login_type"])

        halo_params = halo_params(url="http://localhost:4030",
                                  auth_path="/idsrv/connect/authorize",
                                  login_path="/idsrv/connect/token",
                                  api_path="/graphql",
                                  client_id=None,
                                  client_secret=None,
                                  type="oidc",
                                  login_type="authorization_code",
                                  verify=False, connected=False)

        #file_path = gql_conn.get_info_by_id("http://localhost", "4030", "new.tif",
                                                                    #"tests/unit_tests/test_cred.txt", "test_token")
        file_path = gql_conn.get_info_by_id(halo_params, "new.tif", "http://localhost", "4030",
                                            "tests/unit_tests/test_cred.txt")

        self.assertEqual(file_path, "output_file")


if __name__ == "__main__":
    unittest.main()
