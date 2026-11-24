from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, patch

from deploy.runpod.cost_guard import delete_pod


def test_delete_success_does_not_log_credentials(capsys):
    with patch('deploy.runpod.cost_guard.urlopen') as opened:
        opened.return_value.__enter__.return_value.status = 204
        assert delete_pod('pod123', 'secret')
        request = opened.call_args.args[0]
        assert request.method == 'DELETE'
        assert request.full_url.endswith('/pods/pod123')
    assert capsys.readouterr().out == ''


def test_already_deleted_is_success():
    with patch('deploy.runpod.cost_guard.urlopen', side_effect=HTTPError('url', 404, '', {}, None)):
        assert delete_pod('pod123', 'secret')


def test_network_failure_and_auth_failure_are_not_success():
    for error in [URLError('offline'), HTTPError('url', 403, '', {}, None)]:
        with patch('deploy.runpod.cost_guard.urlopen', side_effect=error):
            assert not delete_pod('pod123', 'secret')
