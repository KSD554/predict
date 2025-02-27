from pymongo import MongoClient
from bcrypt import hashpw, gensalt, checkpw
from flask import session, redirect, url_for
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Connexion à MongoDB
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://sadok:sadok@medi.vdcvl.mongodb.net/?retryWrites=true&w=majority&appName=medi')
client = MongoClient(MONGODB_URI)
db = client['medical_predictions']
users = db['users']
predictions = db['predictions']

def init_db():
    # Création d'index unique pour l'email
    users.create_index('email', unique=True)

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
