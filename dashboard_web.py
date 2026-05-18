from pathlib import Path

from flask import Flask, jsonify, render_template


def create_dashboard_app(monitor):
    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / 'templates'),
        static_folder=str(base_dir / 'static'),
    )

    @app.after_request
    def add_no_cache_headers(resp):
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp

    @app.get('/')
    def dashboard_home():
        return render_template('dashboard.html')

    @app.get('/api/overview')
    def api_overview():
        return jsonify(monitor.get_dashboard_overview_payload())

    @app.get('/api/symbols')
    def api_symbols():
        return jsonify(monitor.get_dashboard_symbols_payload())

    @app.get('/api/positions')
    def api_positions():
        return jsonify(monitor.get_dashboard_positions_payload())

    @app.get('/api/events')
    def api_events():
        return jsonify(monitor.get_dashboard_events_payload(limit=10))

    @app.get('/api/account-stats')
    def api_account_stats():
        return jsonify(monitor.get_dashboard_account_stats_payload())

    @app.get('/api/btc-rr')
    def api_btc_rr():
        return jsonify(monitor.get_dashboard_btc_rr_payload())

    @app.get('/api/position-ai')
    def api_position_ai():
        return jsonify(monitor.get_dashboard_position_ai_payload())

    return app

