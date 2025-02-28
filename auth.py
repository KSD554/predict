from pymongo import MongoClient
from bcrypt import hashpw, gensalt, checkpw
from flask import session, redirect, url_for
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Connexion à MongoDB
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://sadok:sadok@medi.vdcvl.mongodb.net/?retryWrites=true&w=majority&appName=medi')
logger.info(f"Tentative de connexion à MongoDB avec URI: {MONGODB_URI}")

try:
    client = MongoClient(MONGODB_URI)
    # Tester la connexion
    client.admin.command('ping')
    logger.info("Connexion à MongoDB réussie!")
except Exception as e:
    logger.error(f"Erreur de connexion à MongoDB: {str(e)}")
    raise e

db = client['medical_predictions']
users = db['users']
predictions = db['predictions']
blog_posts = db['blog_posts']

def init_db():
    try:
        # Création d'index unique pour l'email
        users.create_index('email', unique=True)
        # Création d'index pour les articles de blog
        blog_posts.create_index([('title', 1)])
        blog_posts.create_index([('created_at', -1)])
        logger.info("Initialisation de la base de données réussie!")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base de données: {str(e)}")
        raise e

def register_user(email, password, name):
    try:
        # Vérifier si les champs sont vides
        if not email or not password or not name:
            return False, "Tous les champs sont obligatoires"
            
        # Vérifier si l'email existe déjà
        if users.find_one({'email': email}):
            return False, "Un compte existe déjà avec cet email"
            
        # Hasher le mot de passe
        hashed = hashpw(password.encode('utf-8'), gensalt())
        
        # Créer l'utilisateur
        user = {
            'email': email,
            'password': hashed,
            'name': name
        }
        
        # Insérer dans la base de données
        result = users.insert_one(user)
        
        # Connecter l'utilisateur automatiquement
        session['user_id'] = str(result.inserted_id)
        session['user_name'] = name
        
        return True, "Inscription réussie"
    except Exception as e:
        return False, f"Erreur lors de l'inscription: {str(e)}"

def login_user(email, password):
    try:
        # Vérifier si les champs sont vides
        if not email or not password:
            return False, "Tous les champs sont obligatoires"
            
        # Trouver l'utilisateur
        user = users.find_one({'email': email})
        
        if not user:
            return False, "Email ou mot de passe incorrect"
            
        if not checkpw(password.encode('utf-8'), user['password']):
            return False, "Email ou mot de passe incorrect"
            
        # Stocker les informations de l'utilisateur dans la session
        session['user_id'] = str(user['_id'])
        session['user_name'] = user['name']
        
        return True, "Connexion réussie"
    except Exception as e:
        return False, f"Erreur lors de la connexion: {str(e)}"

def logout_user():
    try:
        # Supprimer toutes les données de session
        session.clear()
        return True, "Déconnexion réussie"
    except Exception as e:
        return False, f"Erreur lors de la déconnexion: {str(e)}"

def is_authenticated():
    return 'user_id' in session

def get_user_predictions(user_id):
    try:
        return predictions.find({'user_id': user_id})
    except Exception as e:
        return []

def save_prediction(user_id, prediction_data):
    try:
        prediction_data['user_id'] = user_id
        predictions.insert_one(prediction_data)
        return True, "Prédiction sauvegardée avec succès"
    except Exception as e:
        return False, f"Erreur lors de la sauvegarde: {str(e)}"

def create_blog_post(title, content, author_id, author_name, image=None, tags=None):
    try:
        # Vérifier si l'image est fournie
        if not image:
            logger.error("Image manquante pour la création de l'article")
            return False, "L'image de couverture est obligatoire"

        # Vérifier si l'image est en base64
        if not image.startswith('data:image/'):
            logger.error("Format d'image invalide")
            return False, "Format d'image invalide"

        post = {
            'title': title,
            'content': content,
            'author_id': author_id,
            'author_name': author_name,
            'image': image,  # Stockage de l'image en base64
            'tags': tags or [],
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'likes': 0,
            'comments': []
        }
        result = blog_posts.insert_one(post)
        logger.info(f"Article créé avec succès: {result.inserted_id}")
        return True, str(result.inserted_id)
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'article: {str(e)}")
        return False, str(e)

def get_blog_posts(page=1, per_page=10):
    try:
        logger.info(f"Tentative de récupération des articles - Page: {page}, Articles par page: {per_page}")
        skip = (page - 1) * per_page
        total = blog_posts.count_documents({})
        logger.info(f"Nombre total d'articles trouvés: {total}")
        
        posts = list(blog_posts.find().sort('created_at', -1).skip(skip).limit(per_page))
        logger.info(f"Nombre d'articles récupérés pour cette page: {len(posts)}")
        
        # Convertir ObjectId en str pour la sérialisation JSON
        for post in posts:
            post['_id'] = str(post['_id'])
            post['created_at'] = post['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            post['updated_at'] = post['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            'posts': posts,
            'total': total,
            'pages': (total + per_page - 1) // per_page,
            'current_page': page
        }
        logger.info(f"Résultat de la récupération: {result}")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des articles: {str(e)}")
        return None

def get_blog_post(post_id):
    try:
        post = blog_posts.find_one({'_id': ObjectId(post_id)})
        if post:
            post['_id'] = str(post['_id'])
            post['created_at'] = post['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            post['updated_at'] = post['updated_at'].strftime('%Y-%m-%d %H:%M:%S')
        return post
    except Exception as e:
        return None

def update_blog_post(post_id, title, content, image=None, tags=None):
    try:
        update_data = {
            'title': title,
            'content': content,
            'updated_at': datetime.now()
        }
        
        # Mettre à jour l'image si fournie
        if image is not None:
            if not image.startswith('data:image/'):
                logger.error("Format d'image invalide")
                return False
            update_data['image'] = image
            
        if tags is not None:
            update_data['tags'] = tags
            
        result = blog_posts.update_one(
            {'_id': ObjectId(post_id)},
            {'$set': update_data}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de l'article: {str(e)}")
        return False

def delete_blog_post(post_id, author_id):
    try:
        result = blog_posts.delete_one({
            '_id': ObjectId(post_id),
            'author_id': author_id
        })
        return result.deleted_count > 0
    except Exception as e:
        return False

def add_comment(post_id, user_id, user_name, content):
    try:
        comment = {
            'user_id': user_id,
            'user_name': user_name,
            'content': content,
            'created_at': datetime.now()
        }
        result = blog_posts.update_one(
            {'_id': ObjectId(post_id)},
            {'$push': {'comments': comment}}
        )
        return result.modified_count > 0
    except Exception as e:
        return False

def toggle_like(post_id, user_id):
    try:
        # Vérifier si l'utilisateur a déjà liké
        post = blog_posts.find_one({
            '_id': ObjectId(post_id),
            'likes_by': user_id
        })
        
        if post:
            # Unlike
            result = blog_posts.update_one(
                {'_id': ObjectId(post_id)},
                {
                    '$pull': {'likes_by': user_id},
                    '$inc': {'likes': -1}
                }
            )
        else:
            # Like
            result = blog_posts.update_one(
                {'_id': ObjectId(post_id)},
                {
                    '$push': {'likes_by': user_id},
                    '$inc': {'likes': 1}
                }
            )
        return True
    except Exception as e:
        return False
