"""
Module de validation de qualité des données
Ce module implémente une validation personnalisée basée sur Pandas pour vérifier la qualité,
la cohérence et la structure des données météorologiques.
Il remplace l'utilisation précédente de Great Expectations pour une solution plus légère et intégrée.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Validateur de qualité pour les données climatiques"""
    
    def __init__(self):
        self.reports_dir = Path("reports/data_quality")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def create_expectation_suite(self):
        """Création de la suite d'attentes pour les données climatiques"""
        
        expectations = {
            # Attentes générales sur le dataset
            'row_count': {'min_value': 100},  # Au moins 100 lignes
            
            # Attentes sur les colonnes de température
            'temperature_columns': {
                'temperature_2m_mean': {
                    'not_null_percentage': 0.8,  # 80% de valeurs non nulles
                    'min_value': -10,  # Température minimale réaliste pour Marrakech
                    'max_value': 50,   # Température maximale réaliste pour Marrakech
                    'mean_range': [10, 30]  # Moyenne globale attendue
                },
                'temperature_2m_max': {
                    'not_null_percentage': 0.7,
                    'min_value': 0,
                    'max_value': 55
                },
                'temperature_2m_min': {
                    'not_null_percentage': 0.7,
                    'min_value': -10,
                    'max_value': 40
                }
            },
            
            # Attentes sur les dates
            'date_column': {
                'not_null_percentage': 1.0,  # 100% de dates valides
                'date_format': '%Y-%m-%d'
            },
            
            # Attentes sur les features engineered
            'engineered_features': {
                'Year': {'min_value': 2018, 'max_value': 2025},
                'Month': {'min_value': 1, 'max_value': 12},
                'Quarter': {'allowed_values': [1, 2, 3, 4]}
            }
        }
        
        return expectations
    
    def validate_basic_structure(self, df):
        """Validation de la structure de base du dataset"""
        validations = {}
        
        try:
            # Vérification du nombre de lignes
            row_count = len(df)
            validations['row_count'] = {
                'test': 'row_count >= 100',
                'result': row_count >= 100,
                'value': row_count,
                'message': f"Dataset contient {row_count} lignes"
            }
            
            # Vérification des colonnes obligatoires
            # Adaptation pour le dataset Marrakech
            required_columns = ['datetime', 'temperature_2m_mean']
            # Si datetime n'est pas là, vérifier time (format raw)
            if 'datetime' not in df.columns and 'time' in df.columns:
                required_columns = ['time', 'temperature_2m_mean']
                
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            validations['required_columns'] = {
                'test': 'colonnes obligatoires présentes',
                'result': len(missing_columns) == 0,
                'missing_columns': missing_columns,
                'message': f"Colonnes manquantes: {missing_columns}" if missing_columns else "Toutes les colonnes requises sont présentes"
            }
            
            # Vérification des types de données
            type_checks = {}
            date_col = 'datetime' if 'datetime' in df.columns else 'time'
            
            if date_col in df.columns:
                try:
                    pd.to_datetime(df[date_col].iloc[0])
                    type_checks[date_col] = {'valid': True, 'type': 'datetime'}
                except:
                    type_checks[date_col] = {'valid': False, 'type': 'invalid'}
            
            validations['data_types'] = {
                'test': 'types de données corrects',
                'result': all(check['valid'] for check in type_checks.values()),
                'details': type_checks
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur validation structure: {e}")
            validations['structure_error'] = {
                'test': 'validation structure',
                'result': False,
                'error': str(e)
            }
        
        return validations
    
    def validate_temperature_data(self, df):
        """Validation spécifique aux données de température"""
        validations = {}
        
        # Colonnes du dataset Marrakech
        temp_columns = ['temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min']
        
        for col in temp_columns:
            if col not in df.columns:
                continue
                
            try:
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    validations[col] = {
                        'test': f'{col} validation',
                        'result': False,
                        'message': 'Colonne entièrement vide'
                    }
                    continue
                
                # Tests de validité
                tests = {}
                
                # Pourcentage de valeurs non nulles
                null_percentage = df[col].isnull().sum() / len(df)
                tests['completeness'] = {
                    'test': f'non-null rate > 70%',
                    'result': null_percentage < 0.3,
                    'value': f"{(1-null_percentage)*100:.1f}%"
                }
                
                # Valeurs dans une plage réaliste (Marrakech)
                min_val = col_data.min()
                max_val = col_data.max()
                realistic_range = (-10, 55)  # Plage réaliste pour Marrakech
                
                tests['realistic_range'] = {
                    'test': f'températures dans plage réaliste {realistic_range}',
                    'result': min_val >= realistic_range[0] and max_val <= realistic_range[1],
                    'min': min_val,
                    'max': max_val
                }
                
                # Valeurs aberrantes (outliers)
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_percentage = len(outliers) / len(col_data)
                
                tests['outliers'] = {
                    'test': 'outliers < 5%',
                    'result': outlier_percentage < 0.05,
                    'percentage': f"{outlier_percentage*100:.2f}%",
                    'count': len(outliers)
                }
                
                # Moyennes cohérentes (pour la température moyenne)
                if col == 'temperature_2m_mean':
                    mean_temp = col_data.mean()
                    expected_range = (10, 30)  # Moyenne globale attendue pour Marrakech
                    
                    tests['mean_coherence'] = {
                        'test': f'moyenne dans plage attendue {expected_range}',
                        'result': expected_range[0] <= mean_temp <= expected_range[1],
                        'mean': mean_temp
                    }
                
                validations[col] = {
                    'test': f'{col} validation complète',
                    'result': all(test['result'] for test in tests.values()),
                    'details': tests
                }
                
            except Exception as e:
                logger.error(f"❌ Erreur validation {col}: {e}")
                validations[col] = {
                    'test': f'{col} validation',
                    'result': False,
                    'error': str(e)
                }
        
        return validations
    
    def validate_temporal_consistency(self, df):
        """Validation de la cohérence temporelle"""
        validations = {}
        
        try:
            date_col = 'datetime' if 'datetime' in df.columns else 'time'
            
            if date_col not in df.columns:
                return {'temporal': {'test': 'cohérence temporelle', 'result': False, 'message': f'Colonne {date_col} manquante'}}
            
            # Conversion en datetime
            df_temp = df.copy()
            df_temp['dt'] = pd.to_datetime(df_temp[date_col])
            df_temp = df_temp.sort_values('dt')
            
            # Tests temporels
            tests = {}
            
            # Plage de dates cohérente (2018-2023 pour ce dataset)
            min_date = df_temp['dt'].min()
            max_date = df_temp['dt'].max()
            
            tests['date_range'] = {
                'test': 'dates dans plage historique valide',
                'result': min_date.year >= 2018 and max_date.year <= 2025,
                'min_date': min_date.strftime('%Y-%m-%d'),
                'max_date': max_date.strftime('%Y-%m-%d')
            }
            
            # Pas de gaps importants dans les données
            df_temp['date_diff'] = df_temp['dt'].diff()
            large_gaps = df_temp['date_diff'] > pd.Timedelta(days=40)  # Gap > 40 jours
            
            tests['temporal_gaps'] = {
                'test': 'pas de gaps temporels importants',
                'result': large_gaps.sum() < len(df_temp) * 0.01,  # < 1% de gaps
                'large_gaps_count': large_gaps.sum()
            }
            
            # Distribution mensuelle équilibrée
            if len(df_temp) > 12:
                monthly_counts = df_temp.groupby(df_temp['dt'].dt.month).size()
                monthly_balance = monthly_counts.std() / monthly_counts.mean()
                
                tests['monthly_balance'] = {
                    'test': 'distribution mensuelle équilibrée',
                    'result': monthly_balance < 2.0,  # Coefficient de variation < 2
                    'balance_score': monthly_balance
                }
            
            validations['temporal'] = {
                'test': 'cohérence temporelle complète',
                'result': all(test['result'] for test in tests.values()),
                'details': tests
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur validation temporelle: {e}")
            validations['temporal'] = {
                'test': 'cohérence temporelle',
                'result': False,
                'error': str(e)
            }
        
        return validations
    
    def validate_data_quality(self, data_path="data/processed/train.csv"):
        """Validation principale de la qualité des données"""
        try:
            logger.info(f"✅ Démarrage validation qualité: {data_path}")
            
            # Chargement des données
            df = pd.read_csv(data_path)
            logger.info(f"📊 Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Exécution des validations
            all_validations = {}
            
            # 1. Structure de base
            all_validations.update(self.validate_basic_structure(df))
            
            # 2. Données de température
            all_validations.update(self.validate_temperature_data(df))
            
            # 3. Cohérence temporelle
            all_validations.update(self.validate_temporal_consistency(df))
            
            # Calcul du score global
            total_tests = 0
            passed_tests = 0
            
            for validation_group in all_validations.values():
                if isinstance(validation_group.get('result'), bool):
                    total_tests += 1
                    if validation_group['result']:
                        passed_tests += 1
                elif 'details' in validation_group:
                    for detail in validation_group['details'].values():
                        if isinstance(detail.get('result'), bool):
                            total_tests += 1
                            if detail['result']:
                                passed_tests += 1
            
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            is_valid = success_rate >= 0.8  # 80% de tests réussis requis
            
            # Résultats finaux
            results = {
                'is_valid': is_valid,
                'success_rate': success_rate,
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'validations': all_validations,
                'timestamp': datetime.now().isoformat(),
                'data_path': data_path,
                'data_shape': {'rows': len(df), 'columns': len(df.columns)}
            }
            
            # Sauvegarde du rapport
            self.save_validation_report(results)
            
            if is_valid:
                logger.info(f"✅ Validation réussie: {success_rate:.1%} ({passed_tests}/{total_tests} tests)")
            else:
                logger.warning(f"⚠️ Validation échouée: {success_rate:.1%} ({passed_tests}/{total_tests} tests)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la validation: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_validation_report(self, results):
        """Sauvegarde du rapport de validation"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.reports_dir / f"data_quality_report_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Rapport de validation sauvegardé: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde rapport: {e}")

# Fonction principale pour Airflow
def validate_data_quality(data_path="data/processed/train.csv"):
    """Fonction principale appelée par Airflow"""
    validator = DataQualityValidator()
    return validator.validate_data_quality(data_path)

if __name__ == "__main__":
    # Test local
    logging.basicConfig(level=logging.INFO)
    
    validator = DataQualityValidator()
    results = validator.validate_data_quality()
    
    print(f"✅ Validation: {'Réussie' if results['is_valid'] else 'Échouée'}")
    print(f"📊 Score: {results.get('success_rate', 0):.1%}")