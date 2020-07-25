from flask import Flask, render_template
import os

def create_app(test_config: str=None) -> None:
    app = Flask(__name__, instance_relative_config=True)
    
    app.config.from_mapping(SECRET_KEY="dev")
    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.update(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route("/")
    def hello():
        return render_template('index.html')

    app.add_url_rule("/", endpoint="index")
    return app


