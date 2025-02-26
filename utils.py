import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

class PredictionHistory:
    def __init__(self, history_file='prediction_history.json'):
        self.history_file = history_file
        self._ensure_history_file()

    def _ensure_history_file(self):
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump([], f)

    def add_prediction(self, prediction_data):
        predictions = self.get_predictions()
        prediction_data['timestamp'] = datetime.now().isoformat()
        predictions.append(prediction_data)
        
        with open(self.history_file, 'w') as f:
            json.dump(predictions, f)

    def get_predictions(self):
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except:
            return []

    def get_last_prediction(self, disease_type):
        try:
            # Filtrer les prédictions pour le type de maladie donné
            predictions = self.get_predictions()
            disease_predictions = [p for p in predictions if p['disease_type'] == disease_type]
            
            # Trier par timestamp et prendre la plus récente
            if disease_predictions:
                return max(disease_predictions, key=lambda x: datetime.fromisoformat(x['timestamp']))
            
            return None
        except Exception as e:
            print(f"Erreur dans get_last_prediction: {str(e)}")
            return None

    def get_user_statistics(self):
        predictions = self.get_predictions()
        if not predictions:
            return None

        stats = {
            'total_predictions': len(predictions),
            'risk_levels': {
                'Élevé': 0,
                'Modéré': 0,
                'Faible': 0
            },
            'disease_types': {
                'diabetes': 0,
                'hypertension': 0,
                'cardiovascular': 0
            }
        }

        for pred in predictions:
            stats['risk_levels'][pred['risk_level']] += 1
            stats['disease_types'][pred['disease_type']] += 1

        return stats

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  
            textColor=colors.HexColor('#1B4F72')  
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#2E86C1'),
            spaceAfter=20,
            alignment=1
        )
        self.header_style = ParagraphStyle(
            'Header',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#566573'),
            spaceAfter=15
        )

    def create_report(self, prediction_data, output_path):
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []

        # En-tête avec titre
        title = Paragraph(f"Rapport d'Analyse de Santé", self.title_style)
        story.append(title)
        
        # Auteur du rapport
        author = Paragraph("Généré par Dr. Sadok", self.subtitle_style)
        story.append(author)
        story.append(Spacer(1, 10))

        # Informations générales avec style amélioré
        story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", self.header_style))
        story.append(Paragraph(f"Type d'analyse: {prediction_data['disease_type'].capitalize()}", self.header_style))
        story.append(Spacer(1, 20))

        # Niveau de risque
        risk_style = ParagraphStyle(
            'Risk',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=colors.red if prediction_data['risk_level'] == 'Élevé' else
                     colors.orange if prediction_data['risk_level'] == 'Modéré' else
                     colors.green
        )
        story.append(Paragraph(f"Niveau de Risque: {prediction_data['risk_level']}", risk_style))
        story.append(Spacer(1, 20))

        # Facteurs de risque
        story.append(Paragraph("Facteurs de Risque:", self.styles['Heading2']))
        for factor in prediction_data['risk_factors']:
            color = colors.red if factor['severity'] == 'high' else \
                   colors.orange if factor['severity'] == 'medium' else \
                   colors.green
            factor_style = ParagraphStyle(
                'Factor',
                parent=self.styles['Normal'],
                textColor=color
            )
            story.append(Paragraph(f"• {factor['name']}: {factor['description']}", factor_style))
        story.append(Spacer(1, 20))

        # Recommandations
        story.append(Paragraph("Recommandations:", self.styles['Heading2']))
        for rec in prediction_data['recommendations']:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        story.append(Spacer(1, 20))

        # Valeurs mesurées
        story.append(Paragraph("Valeurs Mesurées:", self.styles['Heading2']))
        data = []
        headers = ['Paramètre', 'Valeur', 'Unité']
        data.append(headers)
        
        units = {
            'glucose': 'mg/dL',
            'bmi': 'kg/m²',
            'systolic': 'mmHg',
            'diastolic': 'mmHg',
            'heart_rate': 'bpm',
            'cholesterol': 'mg/dL',
            'age': 'ans'
        }
        
        for key, value in prediction_data['input_data'].items():
            data.append([key.capitalize(), str(value), units.get(key, '-')])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)

        doc.build(story)

    def create_statistics_report(self, history_data, output_path):
        """Génère un rapport statistique basé sur l'historique des prédictions"""
        df = pd.DataFrame(history_data)
        
        # Créer les visualisations
        plt.figure(figsize=(15, 10))
        
        # Distribution des niveaux de risque
        plt.subplot(2, 2, 1)
        risk_counts = df['risk_level'].value_counts()
        colors = ['#ff9999', '#ffd700', '#90EE90']
        plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Distribution des Niveaux de Risque')
        
        # Évolution temporelle
        plt.subplot(2, 2, 2)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        risk_evolution = df.groupby(['date', 'risk_level']).size().unstack(fill_value=0)
        risk_evolution.plot(kind='line', marker='o')
        plt.title('Évolution des Risques dans le Temps')
        plt.xticks(rotation=45)
        
        # Distribution par type de maladie
        plt.subplot(2, 2, 3)
        disease_counts = df['disease_type'].value_counts()
        sns.barplot(x=disease_counts.index, y=disease_counts.values)
        plt.title('Distribution par Type de Maladie')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('temp_stats.png')
        plt.close()
        
        # Créer le rapport PDF
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Titre
        title = Paragraph("Rapport Statistique des Prédictions", self.title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Statistiques générales
        story.append(Paragraph("Statistiques Générales:", self.styles['Heading2']))
        stats_data = [
            ['Nombre total de prédictions', str(len(df))],
            ['Période couverte', f"{df['timestamp'].min().date()} - {df['timestamp'].max().date()}"],
            ['Risque élevé (%)', f"{(df['risk_level'] == 'Élevé').mean()*100:.1f}%"],
            ['Type le plus fréquent', df['disease_type'].mode()[0]]
        ]
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Ajouter le graphique
        story.append(Paragraph("Visualisations:", self.styles['Heading2']))
        story.append(Spacer(1, 10))
        story.append(Image('temp_stats.png', width=450, height=300))
        
        doc.build(story)
        
        # Nettoyer les fichiers temporaires
        os.remove('temp_stats.png')
