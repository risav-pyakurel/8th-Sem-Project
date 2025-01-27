from flask import Blueprint, request, jsonify
from models import db, News, SummarizedNews, Category

routes = Blueprint('routes', __name__)

@routes.route('/categories', methods=['POST'])
def create_category():
    data = request.get_json()
    new_category = Category(name=data['name'])
    db.session.add(new_category)
    db.session.commit()
    return jsonify({'message': 'Category created successfully'}), 201

@routes.route('/news', methods=['POST'])
def create_news():
    data = request.get_json()
    new_news = News(title=data['title'], content=data['content'], category_id=data['category_id'])
    db.session.add(new_news)
    db.session.commit()
    return jsonify({'message': 'News created successfully'}), 201

@routes.route('/summarized_news', methods=['POST'])
def create_summarized_news():
    data = request.get_json()
    new_summarized_news = SummarizedNews(summary=data['summary'], news_id=data['news_id'], category_id=data['category_id'])
    db.session.add(new_summarized_news)
    db.session.commit()
    return jsonify({'message': 'Summarized News created successfully'}), 201

# get category
@routes.route('/categories', methods=['GET'])
def get_categories():
    category_list = Category.query.all()
    return jsonify([{'id': category.id, 'name': category.name} for category in category_list]), 200


@routes.route('/news', methods=['GET'])
def get_news():
    news_list = News.query.all()
    return jsonify([{'id': news.id, 'title': news.title, 'content': news.content, 'category_id': news.category_id} for news in news_list]), 200

@routes.route('/summarized_news', methods=['GET'])
def get_summarized_news():
    summarized_news_list = SummarizedNews.query.all()
    return jsonify([{'id': summarized_news.id, 'summary': summarized_news.summary, 'news_id': summarized_news.news_id, 'category_id': summarized_news.category_id} for summarized_news in summarized_news_list]), 200