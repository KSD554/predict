from flask import Flask, render_template, request, jsonify, send_file, make_response, redirect, url_for, session
from dotenv import load_dotenv
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils import PredictionHistory, ReportGenerator
import tempfile
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from flask_session import Session
from auth import (
    init_db, register_user, login_user, logout_user, is_authenticated, save_prediction,
    client, create_blog_post, get_blog_posts, get_blog_post, update_blog_post,
    delete_blog_post, add_comment, toggle_like
)

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'votre_cle_secrete_tres_longue_et_aleatoire')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'

# Assurez-vous que le répertoire de session existe
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

Session(app)

# Initialiser la base de données MongoDB
init_db()

# Initialiser les gestionnaires
prediction_history = PredictionHistory()
report_generator = ReportGenerator()

# Initialiser les modèles
def initialize_models():
    # Créer des modèles simples pour la démonstration
    models = {}
    scalers = {}
    
    for disease in ['diabetes', 'hypertension', 'cardiovascular']:
        models[disease] = RandomForestClassifier(n_estimators=100, random_state=42)
        scalers[disease] = StandardScaler()
        
        # Générer des données synthétiques pour l'entraînement
        np.random.seed(42)  # Pour la reproductibilité
        X = np.random.normal(size=(1000, 3))  # Distribution normale pour plus de réalisme
        
        if disease == 'diabetes':
            # Glucose (70-400), Age (18-120), BMI (15-50)
            X[:, 0] = X[:, 0] * 50 + 150  # glucose moyen = 150, écart-type = 50
            X[:, 1] = X[:, 1] * 20 + 50   # age moyen = 50, écart-type = 20
            X[:, 2] = X[:, 2] * 5 + 25    # BMI moyen = 25, écart-type = 5
            
            # Règles de classification plus réalistes pour le diabète
            y = ((X[:, 0] > 200) |  # glycémie élevée
                 ((X[:, 2] > 30) & (X[:, 1] > 45)) |  # BMI élevé et âge > 45
                 ((X[:, 0] > 150) & (X[:, 2] > 27)))  # glycémie modérée et surpoids
                
        elif disease == 'hypertension':
            # Systolic (90-200), Diastolic (60-120), Age (18-120)
            X[:, 0] = X[:, 0] * 20 + 130  # systolic moyen = 130, écart-type = 20
            X[:, 1] = X[:, 1] * 10 + 80   # diastolic moyen = 80, écart-type = 10
            X[:, 2] = X[:, 1] * 20 + 50   # age moyen = 50, écart-type = 20
            
            # Règles de classification plus réalistes pour l'hypertension
            y = ((X[:, 0] > 140) & (X[:, 1] > 90) |  # hypertension classique
                 (X[:, 0] > 160) |  # systolique très élevée
                 (X[:, 1] > 100) |  # diastolique très élevée
                 ((X[:, 0] > 130) & (X[:, 1] > 85) & (X[:, 2] > 60)))  # pré-hypertension avec âge
                
        else:  # cardiovascular
            # Heart rate (40-120), Cholesterol (100-300), Age (18-120)
            X[:, 0] = X[:, 0] * 15 + 75   # heart rate moyen = 75, écart-type = 15
            X[:, 1] = X[:, 1] * 40 + 200  # cholesterol moyen = 200, écart-type = 40
            X[:, 2] = X[:, 1] * 20 + 50   # age moyen = 50, écart-type = 20
            
            # Règles de classification plus réalistes pour les maladies cardiovasculaires
            y = ((X[:, 1] > 240) |  # cholestérol élevé
                 ((X[:, 0] > 100) & (X[:, 2] > 60)) |  # fréquence cardiaque élevée et âge
                 ((X[:, 1] > 200) & (X[:, 2] > 55)) |  # cholestérol modéré et âge
                 ((X[:, 0] < 50) & (X[:, 2] > 40)))  # bradycardie et âge
        
        # Normaliser et entraîner
        X_scaled = scalers[disease].fit_transform(X)
        models[disease].fit(X_scaled, y)
    
    return models, scalers

# Initialiser les modèles au démarrage
models, scalers = initialize_models()

@app.route('/')
def index():
    initial_data = {
        'notification': {
            'show': False,
            'type': 'success',
            'message': ''
        },
        'prediction': None,
        'is_authenticated': str(is_authenticated()).lower(),
        'user_name': session.get('user_name', '')
    }
    return render_template('index.html', **initial_data)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    success, message = register_user(data['email'], data['password'], data['name'])
    return jsonify({'success': success, 'message': message})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    success, message = login_user(data['email'], data['password'])
    return jsonify({'success': success, 'message': message})

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

def get_risk_factors(disease_type, input_data, prediction):
    risk_factors = []
    
    if disease_type == 'diabetes':
        # Analyse de la glycémie
        glucose = float(input_data['glucose'])
        if glucose >= 200:
            risk_factors.append({
                'name': 'Glycémie très élevée',
                'severity': 'high',
                'description': 'Votre taux de glucose ({}mg/dL) est significativement au-dessus de la normale. Un taux supérieur à 200mg/dL peut indiquer un diabète. Une consultation médicale est recommandée.'.format(int(glucose))
            })
        elif glucose >= 140:
            risk_factors.append({
                'name': 'Glycémie élevée',
                'severity': 'medium',
                'description': 'Votre taux de glucose ({}mg/dL) est supérieur à la normale. Un taux entre 140 et 199mg/dL peut indiquer un pré-diabète.'.format(int(glucose))
            })
        
        # Analyse de l'IMC
        bmi = float(input_data['bmi'])
        if bmi >= 30:
            risk_factors.append({
                'name': 'Obésité',
                'severity': 'high',
                'description': 'Votre IMC de {} indique une obésité. L\'obésité augmente significativement le risque de diabète et d\'autres problèmes de santé.'.format(round(bmi, 1))
            })
        elif bmi >= 25:
            risk_factors.append({
                'name': 'Surpoids',
                'severity': 'medium',
                'description': 'Votre IMC de {} indique un surpoids. Le surpoids peut augmenter le risque de diabète. Une perte de poids modérée pourrait réduire ce risque.'.format(round(bmi, 1))
            })

        # Analyse de l'âge
        age = float(input_data['age'])
        if age >= 45:
            risk_factors.append({
                'name': 'Âge à risque',
                'severity': 'medium',
                'description': 'À {} ans, vous faites partie d\'une tranche d\'âge où le risque de diabète est plus élevé. Un dépistage régulier est recommandé.'.format(int(age))
            })

    elif disease_type == 'hypertension':
        # Analyse de la pression systolique
        systolic = float(input_data['systolic'])
        diastolic = float(input_data['diastolic'])
        
        if systolic >= 160 or diastolic >= 100:
            risk_factors.append({
                'name': 'Hypertension sévère',
                'severity': 'high',
                'description': 'Votre pression artérielle ({}/{}mmHg) indique une hypertension sévère. Une consultation médicale urgente est nécessaire.'.format(int(systolic), int(diastolic))
            })
        elif systolic >= 140 or diastolic >= 90:
            risk_factors.append({
                'name': 'Hypertension',
                'severity': 'medium',
                'description': 'Votre pression artérielle ({}/{}mmHg) indique une hypertension. Un suivi médical régulier est recommandé.'.format(int(systolic), int(diastolic))
            })
        elif systolic >= 130 or diastolic >= 85:
            risk_factors.append({
                'name': 'Pré-hypertension',
                'severity': 'low',
                'description': 'Votre pression artérielle ({}/{}mmHg) est légèrement élevée. Des modifications du mode de vie pourraient aider à la réduire.'.format(int(systolic), int(diastolic))
            })

    else:  # cardiovascular
        # Analyse du cholestérol
        cholesterol = float(input_data['cholesterol'])
        if cholesterol >= 240:
            risk_factors.append({
                'name': 'Cholestérol très élevé',
                'severity': 'high',
                'description': 'Votre taux de cholestérol ({}mg/dL) est très élevé. Un taux élevé augmente significativement le risque de maladies cardiovasculaires.'.format(int(cholesterol))
            })
        elif cholesterol >= 200:
            risk_factors.append({
                'name': 'Cholestérol élevé',
                'severity': 'medium',
                'description': 'Votre taux de cholestérol ({}mg/dL) est à la limite supérieure. Une surveillance et des modifications alimentaires sont recommandées.'.format(int(cholesterol))
            })

        # Analyse de la fréquence cardiaque
        heart_rate = float(input_data['heart_rate'])
        if heart_rate >= 100:
            risk_factors.append({
                'name': 'Fréquence cardiaque élevée',
                'severity': 'medium',
                'description': 'Votre fréquence cardiaque ({}bpm) est élevée. Une fréquence élevée au repos peut indiquer un risque cardiovasculaire.'.format(int(heart_rate))
            })
        elif heart_rate < 60:
            risk_factors.append({
                'name': 'Fréquence cardiaque basse',
                'severity': 'low',
                'description': 'Votre fréquence cardiaque ({}bpm) est basse. Si vous n\'êtes pas un athlète, cela pourrait nécessiter une évaluation.'.format(int(heart_rate))
            })

    return risk_factors

def get_recommendations(disease_type, risk_level):
    recommendations = []
    
    if disease_type == 'diabetes':
        if risk_level == 'Faible':
            recommendations.append({
                'title': 'Recommandations générales',
                'items': [
                    "Maintenir une alimentation équilibrée",
                    "Faire de l'exercice régulièrement",
                    "Surveiller votre glycémie occasionnellement"
                ]
            })
        elif risk_level == 'Modéré':
            recommendations.append({
                'title': 'Surveillance et mode de vie',
                'items': [
                    "Réduire la consommation de sucres raffinés",
                    "Augmenter l'activité physique à 30 minutes par jour",
                    "Surveiller votre glycémie hebdomadairement",
                    "Consulter un nutritionniste"
                ]
            })
        else:  # Élevé
            recommendations.append({
                'title': 'Actions urgentes',
                'items': [
                    "Consulter un médecin rapidement",
                    "Surveiller votre glycémie quotidiennement",
                    "Suivre un régime strict",
                    "Maintenir un journal alimentaire"
                ]
            })
    
    elif disease_type == 'hypertension':
        if risk_level == 'Faible':
            recommendations.append({
                'title': 'Prévention',
                'items': [
                    "Maintenir une alimentation pauvre en sel",
                    "Pratiquer une activité physique régulière",
                    "Gérer votre stress"
                ]
            })
        elif risk_level == 'Modéré':
            recommendations.append({
                'title': 'Mode de vie à adapter',
                'items': [
                    "Réduire significativement la consommation de sel",
                    "Exercice cardiovasculaire 3 fois par semaine",
                    "Éviter l'alcool et le tabac",
                    "Pratiquer la méditation"
                ]
            })
        else:  # Élevé
            recommendations.append({
                'title': 'Suivi médical',
                'items': [
                    "Consulter un cardiologue rapidement",
                    "Suivre un régime DASH",
                    "Surveiller votre tension quotidiennement",
                    "Limiter la caféine"
                ]
            })
    
    elif disease_type == 'cardiovascular':
        if risk_level == 'Faible':
            recommendations.append({
                'title': 'Prévention cardiovasculaire',
                'items': [
                    "Maintenir un mode de vie actif",
                    "Suivre une alimentation équilibrée",
                    "Éviter le tabac"
                ]
            })
        elif risk_level == 'Modéré':
            recommendations.append({
                'title': 'Surveillance cardiovasculaire',
                'items': [
                    "Augmenter l'activité physique",
                    "Adopter un régime méditerranéen",
                    "Gérer le stress",
                    "Surveiller le cholestérol"
                ]
            })
        else:  # Élevé
            recommendations.append({
                'title': 'Actions immédiates',
                'items': [
                    "Consulter un cardiologue d'urgence",
                    "Suivre un programme de réadaptation cardiaque",
                    "Contrôler strictement le cholestérol et la tension",
                    "Arrêter immédiatement le tabac"
                ]
            })
    
    # Ajouter des recommandations générales pour tous les niveaux
    recommendations.append({
        'title': 'Recommandations générales de santé',
        'items': [
            "Maintenir un poids santé",
            "Dormir suffisamment (7-8 heures par nuit)",
            "Rester hydraté",
            "Gérer votre stress au quotidien"
        ]
    })
    
    return recommendations

@app.route('/predict', methods=['POST'])
def predict():
    # Vérification détaillée de l'authentification
    app.logger.info(f"Session actuelle: {session}")
    app.logger.info(f"État d'authentification: {is_authenticated()}")
    app.logger.info(f"User ID dans la session: {session.get('user_id')}")
    
    if not is_authenticated():
        app.logger.error("Utilisateur non authentifié")
        return jsonify({
            'success': False,
            'message': 'Veuillez vous connecter pour faire une prédiction'
        }), 401

    try:
        data = request.get_json()
        if not data:
            app.logger.error("Aucune donnée reçue")
            return jsonify({
                'success': False,
                'message': 'Aucune donnée reçue'
            }), 400

        app.logger.info(f"Données brutes reçues: {data}")
        app.logger.info(f"Type des données reçues: {type(data)}")
        
        disease_type = data.get('disease_type')
        input_data = data.get('input_data')
        
        if not disease_type or not input_data:
            app.logger.error(f"Données manquantes: disease_type={disease_type}, input_data={input_data}")
            return jsonify({
                'success': False,
                'message': 'Données manquantes: type de maladie et données requises'
            }), 400

        # Vérification du type de maladie
        if disease_type not in ['diabetes', 'hypertension', 'cardiovascular']:
            return jsonify({
                'success': False,
                'message': 'Type de maladie non valide'
            }), 400

        # Vérifier le type des données d'entrée
        app.logger.info(f"Type des données d'entrée: {type(input_data)}")
        app.logger.info(f"Contenu des données d'entrée: {input_data}")
        app.logger.info(f"Clés dans input_data: {input_data.keys() if isinstance(input_data, dict) else 'Non dictionnaire'}")

        # Préparer les données pour la prédiction
        if disease_type == 'diabetes':
            X = np.array([[
                float(input_data['glucose']),
                float(input_data['age']),
                float(input_data['bmi'])
            ]])
        elif disease_type == 'hypertension':
            X = np.array([[
                float(input_data['systolic']),
                float(input_data['diastolic']),
                float(input_data['age'])
            ]])
        else:  # cardiovascular
            X = np.array([[
                float(input_data['heart_rate']),
                float(input_data['cholesterol']),
                float(input_data['age'])
            ]])

        # Normaliser les données
        X_scaled = scalers[disease_type].transform(X)
        
        # Faire la prédiction
        prediction = bool(models[disease_type].predict(X_scaled)[0])
        
        # Obtenir les facteurs de risque
        risk_factors = get_risk_factors(disease_type, input_data, prediction)
        
        # Obtenir les recommandations
        risk_level = 'Élevé' if prediction else 'Faible'
        recommendations = get_recommendations(disease_type, risk_level)
        
        # Créer l'objet de réponse
        response_data = {
            'success': True,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'probability': 75 if prediction else 25  # Valeur exemple
        }
        
        # Sauvegarder la prédiction dans MongoDB
        prediction_data = {
            'disease_type': disease_type,
            'input_data': input_data,
            'prediction': prediction,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
        save_prediction(session['user_id'], prediction_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f'Erreur lors de la prédiction: {str(e)}')
        return jsonify({
            'success': False,
            'message': f'Erreur lors de la prédiction: {str(e)}'
        }), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        
        # Créer un fichier temporaire pour le rapport
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            report_path = tmp.name
            report_generator.create_report(data, report_path)
            
            return send_file(
                report_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name='rapport_sante.pdf'
            )
    except Exception as e:
        app.logger.error(f'Erreur lors de la génération du rapport : {str(e)}')
        return jsonify({
            'error': 'Erreur lors de la génération du rapport'
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    if not is_authenticated():
        return jsonify({
            'error': 'Veuillez vous connecter pour accéder à l\'historique'
        }), 401
        
    try:
        predictions = prediction_history.get_predictions()
        return jsonify(predictions)
    except Exception as e:
        app.logger.error(f'Erreur lors de la récupération de l\'historique : {str(e)}')
        return jsonify({
            'error': 'Erreur lors de la récupération de l\'historique'
        }), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    if not is_authenticated():
        return jsonify({
            'error': 'Veuillez vous connecter pour accéder aux statistiques'
        }), 401
        
    try:
        stats = prediction_history.get_user_statistics()
        if not stats:
            return jsonify({
                'error': 'Aucune donnée disponible pour les statistiques'
            }), 404
            
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f'Erreur lors de la récupération des statistiques : {str(e)}')
        return jsonify({
            'error': 'Erreur lors de la récupération des statistiques'
        }), 500

@app.route('/statistics_report', methods=['GET'])
def generate_statistics_report():
    if not is_authenticated():
        return jsonify({
            'error': 'Veuillez vous connecter pour générer le rapport'
        }), 401
        
    try:
        predictions = prediction_history.get_predictions()
        if not predictions:
            return jsonify({
                'error': 'Aucune donnée disponible pour le rapport statistique'
            }), 404
            
        # Créer un fichier temporaire pour le rapport
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            report_path = tmp.name
            report_generator.create_statistics_report(predictions, report_path)
            
            return send_file(
                report_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name='rapport_statistique.pdf'
            )
    except Exception as e:
        app.logger.error(f'Erreur lors de la génération du rapport statistique : {str(e)}')
        return jsonify({
            'error': 'Erreur lors de la génération du rapport statistique'
        }), 500

@app.route('/download_report/<disease_type>')
def download_report(disease_type):
    try:
        print(f"Tentative de téléchargement pour {disease_type}")
        # Récupérer la dernière prédiction
        prediction = prediction_history.get_last_prediction(disease_type)
        print(f"Prédiction trouvée: {prediction}")
        
        if not prediction:
            print("Aucune prédiction trouvée")
            return jsonify({'error': 'Aucune prédiction trouvée'}), 404

        # Créer un buffer pour le PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Créer un style personnalisé pour le titre
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1E40AF')  # Bleu foncé
        )
        
        # Créer un style pour le texte normal
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.HexColor('#374151')  # Gris foncé
        )
        
        # Créer un style pour les sous-titres
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#1E40AF')
        )

        # Préparer le contenu
        elements = []
        
        # Titre
        elements.append(Paragraph(f"Rapport de Prédiction de Risque - {disease_type.capitalize()}", title_style))
        elements.append(Spacer(1, 12))
        
        # Date
        elements.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
        elements.append(Spacer(1, 12))
        
        # Niveau de risque
        elements.append(Paragraph("Résultats", subtitle_style))
        risk_level = prediction.get('risk_level', 'Non disponible')
        probability = prediction.get('probability', 0)
        if isinstance(probability, (int, float)):
            probability = f"{probability:.1f}"
        elements.append(Paragraph(f"Niveau de Risque: {risk_level}", normal_style))
        elements.append(Paragraph(f"Probabilité: {probability}%", normal_style))
        elements.append(Spacer(1, 12))
        
        # Description
        if prediction.get('risk_description'):
            elements.append(Paragraph("Description", subtitle_style))
            elements.append(Paragraph(prediction['risk_description'], normal_style))
            elements.append(Spacer(1, 12))
        
        # Facteurs de risque
        risk_factors = prediction.get('risk_factors', [])
        if risk_factors:
            elements.append(Paragraph("Facteurs de Risque", subtitle_style))
            factors_list = []
            for factor in risk_factors:
                if isinstance(factor, dict):
                    text = f"{factor.get('name', '')} - {factor.get('description', '')}"
                else:
                    text = str(factor)
                factors_list.append(ListItem(Paragraph(text, normal_style)))
            if factors_list:
                elements.append(ListFlowable(factors_list, bulletType='bullet'))
                elements.append(Spacer(1, 12))
        
        # Recommandations
        recommendations = prediction.get('recommendations', [])
        if recommendations:
            elements.append(Paragraph("Recommandations", subtitle_style))
            for rec in recommendations:
                if isinstance(rec, dict):
                    title = rec.get('title', '')
                    if title:
                        elements.append(Paragraph(title, subtitle_style))
                    items = []
                    for item in rec.get('items', []):
                        items.append(ListItem(Paragraph(str(item), normal_style)))
                    if items:
                        elements.append(ListFlowable(items, bulletType='bullet'))
                        elements.append(Spacer(1, 12))
                else:
                    elements.append(Paragraph(str(rec), normal_style))
                    elements.append(Spacer(1, 12))

        # Générer le PDF
        doc.build(elements)
        
        # Préparer la réponse
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=rapport_{disease_type}_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        return response

    except Exception as e:
        print(f"Erreur lors de la génération du PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Erreur lors de la génération du rapport'}), 500

@app.route('/blog')
def blog():
    page = request.args.get('page', 1, type=int)
    result = get_blog_posts(page=page)
    if result is None:
        return render_template('blog.html', posts=[], total=0, pages=1, current_page=1)
    return render_template('blog.html', **result)

@app.route('/blog/post/<post_id>')
def view_post(post_id):
    post = get_blog_post(post_id)
    if not post:
        return redirect(url_for('blog'))
    return render_template('blog_post.html', post=post)

@app.route('/blog/new', methods=['GET', 'POST'])
def new_post():
    if not is_authenticated():
        return redirect(url_for('blog'))
        
    if request.method == 'POST':
        data = request.get_json()
        success, post_id = create_blog_post(
            title=data['title'],
            content=data['content'],
            author_id=session['user_id'],
            author_name=session['user_name'],
            image_url=data.get('image_url'),
            tags=data.get('tags', [])
        )
        return jsonify({'success': success, 'post_id': post_id})
    
    return render_template('blog_editor.html')

@app.route('/blog/edit/<post_id>', methods=['GET', 'POST'])
def edit_post(post_id):
    if not is_authenticated():
        return redirect(url_for('blog'))
        
    post = get_blog_post(post_id)
    if not post or post['author_id'] != session['user_id']:
        return redirect(url_for('blog'))
        
    if request.method == 'POST':
        data = request.get_json()
        success = update_blog_post(
            post_id=post_id,
            title=data['title'],
            content=data['content'],
            image_url=data.get('image_url'),
            tags=data.get('tags', [])
        )
        return jsonify({'success': success})
    
    return render_template('blog_editor.html', post=post)

@app.route('/blog/delete/<post_id>', methods=['POST'])
def delete_post(post_id):
    if not is_authenticated():
        return jsonify({'success': False, 'message': 'Non autorisé'}), 401
        
    success = delete_blog_post(post_id, session['user_id'])
    return jsonify({'success': success})

@app.route('/blog/comment/<post_id>', methods=['POST'])
def comment_post(post_id):
    if not is_authenticated():
        return jsonify({'success': False, 'message': 'Non autorisé'}), 401
        
    data = request.get_json()
    success = add_comment(
        post_id=post_id,
        user_id=session['user_id'],
        user_name=session['user_name'],
        content=data['content']
    )
    return jsonify({'success': success})

@app.route('/blog/like/<post_id>', methods=['POST'])
def like_post(post_id):
    if not is_authenticated():
        return jsonify({'success': False, 'message': 'Non autorisé'}), 401
        
    success = toggle_like(post_id, session['user_id'])
    return jsonify({'success': success})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
