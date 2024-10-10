from flask import Flask
from app.api.routes import api_bp


def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "Welcome to MedAI-Stroke API"

    app.register_blueprint(api_bp, url_prefix='/api')



    return app