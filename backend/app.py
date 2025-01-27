from flask import Flask
from models import db
from routes import routes
app = Flask(__name__)
app.config.from_object('config.Config')
db.init_app(app)

with app.app_context():
    db.create_all()

app.register_blueprint(routes)

if __name__ == '__main__':
    app.run(debug=True)