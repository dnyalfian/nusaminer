from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
import os
import pandas as pd
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
import io
import base64
import psycopg2
from psycopg2.extras import RealDictCursor


app = Flask(__name__)
CORS(app)

label_encoder_target = LabelEncoder()


# Database Connection
def get_db_connection():
    return psycopg2.connect(
        host="db.pclcdbmhklibarghcjdk.supabase.co",
        user="postgres",
        password="P@ssw0rd",
        database="postgres",
        port=5432,
        sslmode="require",
        cursor_factory=RealDictCursor  # ✅ penting!
    )

@app.route("/test_db")
def test_db():
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({"status": "success", "message": "PostgreSQL connected ✅"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Endpoint: Register
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        name = data.get('name', '')
        phone = data.get('phone', '')
        address = data.get('address', '')
        
        if not username or not email or not password:
            return jsonify({"status": "error", "message": "Username, email, and password are required."})
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s OR username = %s", (email, username))
            existing_user = cursor.fetchone()
            if existing_user:
                return jsonify({"status": "error", "message": "Email or username already registered."})
            
            cursor.execute("INSERT INTO users (username, email, password, name, phone, address) VALUES (%s, %s, %s, %s, %s, %s)", 
                           (username, email, hashed_password, name, phone, address))
            conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "User registered successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Endpoint: Login
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"success": False, "message": "Username dan password harus diisi."})

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id, password FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
        conn.close()

        if not user or not check_password_hash(user['password'], password):
            return jsonify({"success": False, "message": "Username atau password salah."})

        return jsonify({"success": True, "message": "Login berhasil.", "user_id": user['user_id']})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Endpoint: Read User Data
@app.route('/user/profile/<user_id>', methods=['GET'])
def get_profile(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, email, address, phone FROM users WHERE user_id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            return jsonify({'success': True, 'user_data': user})
        else:
            return jsonify({'success': False, 'message': 'User not found'}), 404
    except Exception as e:
        print(f"Error in get_profile: {e}")
        return jsonify({'success': False, 'message': 'Internal Server Error'}), 500

# Endpoint Update User
@app.route('/user/update_profile', methods=['PUT'])
def update_profile():
    try:
        data = request.json
        user_id = data.get('user_id')
        name = data.get('name')
        email = data.get('email')
        address = data.get('address')
        phone = data.get('phone')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET name = %s, email = %s, address = %s, phone = %s WHERE user_id = %s
        """, (name, email, address, phone, user_id))
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Profile updated successfully'})
    except Exception as e:
        print(f"Error in update_profile: {e}")
        return jsonify({'success': False, 'message': 'Internal Server Error'}), 500


# Allowed File Types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Save processed file
def save_processed_file(df, original_filename):
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], original_filename)
    df.to_csv(processed_filepath, index=False)
    return processed_filepath

# Folder upload
UPLOAD_FOLDER = r'C:\xampp\htdocs\p_skripsi\uploads'  
ALLOWED_EXTENSIONS = {'csv', 'txt'}
PROCESSED_FOLDER = r'C:\xampp\htdocs\p_skripsi\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Endpoint: Upload File
@app.route('/upload', methods=['POST'])
def upload_file():
    user_id = request.args.get('user_id')
    if 'file' not in request.files or not user_id:
        return jsonify({"status": "error", "message": "Missing file or user_id"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected for uploading."})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(user_folder, exist_ok=True)
        filepath = os.path.join(user_folder, filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                return jsonify({"status": "error", "message": "Uploaded file is empty."})

            df.to_csv(filepath, index=False)

            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO uploaded_files (user_id, file_name, file_path) VALUES (%s, %s, %s)",
                    (user_id, filename, filepath)
                )
                conn.commit()
            conn.close()

            return jsonify({"status": "success", "message": "File uploaded and saved successfully."})

        except Exception as e:
            return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"})

    else:
        return jsonify({"status": "error", "message": "Invalid file type."})

    
# Endpoint: Columns Dataset (Prediction)
@app.route('/columns', methods=['GET'])
def get_columns():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"status": "error", "message": "User ID is required."}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM uploaded_files WHERE user_id = %s ORDER BY uploaded_at DESC LIMIT 1",
                (user_id,)
            )
            file_record = cursor.fetchone()
        conn.close()

        if not file_record:
            return jsonify({"status": "error", "message": "No files found in the database."})

        file_path = file_record['file_path']
        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": f"File not found: {file_path}"})

        df = pd.read_csv(file_path)
        if df.empty:
            return jsonify({"status": "error", "message": "Dataset is empty."})

        columns = list(df.columns)

        # Get target column from meta or default
        target_column = get_target_column_from_meta(file_path)
        if target_column not in columns:
            target_column = columns[-1]  # fallback if not found

        return jsonify({
            "status": "success",
            "columns": columns,
            "target_column": target_column,
            "message": "Columns retrieved successfully."
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})



# Endpoint: Process Data
@app.route('/process', methods=['GET'])
def process_data():
    try:
        user_id = request.args.get('user_id') 
        action = request.args.get('action')
        print(f" ACTION received: {action}")

        if not user_id:
            return jsonify({"status": "error", "message": "User ID is required."}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM uploaded_files WHERE user_id = %s ORDER BY uploaded_at DESC LIMIT 1",
                (user_id,)
            )
            file_record = cursor.fetchone()
        conn.close()

        if not file_record:
            return jsonify({"status": "error", "message": "No files found for this user."})

        file_path = file_record['file_path']
        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": f"File not found: {file_path}"})

        df = pd.read_csv(file_path)
        if df.empty:
            return jsonify({"status": "error", "message": "Dataset is empty."})

        # Cek jika target label sudah diset sebelumnya
        target_column = get_target_column_from_meta(file_path)

        if not action or action == 'preview':
            return jsonify({
        "status": "success",
        "columns": list(df.columns),
        "head": df.head().to_dict(orient='records'),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "message": "Data preview only (no action applied)"
    })

        if action == 'clean data':
            df = clean_data(df)

        elif action == 'drop columns':
            columns_to_drop = request.args.get('columns', '')
            if columns_to_drop:
                df = drop_columns(df, columns_to_drop)
            else:
                return jsonify({"status": "error", "message": "No columns provided for drop."})

        elif action == 'convert categorical':
            df = convert_categorical(df, target_column=target_column)

        elif action == 'normalize':
            df = normalize_data(df, target_column=target_column)

        elif action == 'set role':
            # Ambil target column dari parameter
            new_target = request.args.get('target_column')
            if not new_target:
                return jsonify({"status": "error", "message": "No target column specified."}), 400
            if new_target not in df.columns:
                return jsonify({"status": "error", "message": f"Target column '{new_target}' not found in dataset."}), 400

            meta_path = file_path.replace('.csv', '_meta.txt')
            with open(meta_path, 'w') as f:
                f.write(new_target)

            return jsonify({
                "status": "success",
                "target_column": new_target,
                "message": f"Kolom '{new_target}' berhasil diset sebagai label (target)."
            })

        else:
            return jsonify({"status": "error", "message": f"Unknown action: {action}"}), 400

        # Simpan hasil perubahan
        df.to_csv(file_path, index=False)

        return jsonify({
            "status": "success",
            "columns": list(df.columns),
            "head": df.head().to_dict(orient='records'),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "message": f"Action '{action}' applied successfully."
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Function to get target column from meta file (or default)
def get_target_column_from_meta(file_path):
    meta_path = file_path.replace('.csv', '_meta.txt')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return f.read().strip()
    return 'Level'  # Default jika tidak ditentukan

# Function to clean data (remove NaN or missing values)
def clean_data(df):
    return df.dropna()

# Function to convert categorical columns to numeric
def convert_categorical(df, target_column='Level'):
    label_encoder = LabelEncoder()
    for column in df.columns:
        if column != target_column and df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])
    return df

# Function to normalize numerical data
def normalize_data(df, target_column='Level'):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    cols_to_normalize = [col for col in numeric_cols if df[col].nunique() > 2]
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    df[cols_to_normalize] = df[cols_to_normalize].round(2)
    return df

# Function to drop specific columns
def drop_columns(df, columns):
    columns_list = columns.split(',')
    return df.drop(columns=columns_list, axis=1, errors='ignore')


# Endpoint: Download
@app.route('/download_split/<filename>', methods=['GET'])
def download_split_file(filename):
    split_dir = os.path.join(os.getcwd(), 'processed', 'split')
    try:
        return send_from_directory(split_dir, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

# def Model:
def build_model(model_type, request):
    if model_type == 'AdaBoost':
        return AdaBoostClassifier(
            n_estimators=request.args.get('n_estimators', 50, type=int),
            learning_rate=request.args.get('learning_rate', 1.0, type=float),
            random_state=42
        )
    elif model_type == 'Artificial Neural Network':
        hidden_layer_sizes = tuple(map(int, request.args.get('hidden_layers', '50,50').split(',')))
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=request.args.get('max_iter', 500, type=int),
            activation='relu',
            solver='adam',
            early_stopping=True,
            random_state=42
        ))
    elif model_type == 'Decision Tree':
        return DecisionTreeClassifier(
            max_depth=request.args.get('max_depth', 5, type=int),
            min_samples_split=request.args.get('min_samples_split', 2, type=int),
            criterion=request.args.get('criterion', 'gini'),
            random_state=42
        )
    elif model_type == 'Gradient Boosting':
        return GradientBoostingClassifier(
            n_estimators=request.args.get('n_estimators', 100, type=int),
            learning_rate=request.args.get('learning_rate', 0.1, type=float),
            max_depth=request.args.get('max_depth', 3, type=int),
            random_state=42
        )
    elif model_type == 'K-Nearest Neighbors':
        return KNeighborsClassifier(
            n_neighbors=request.args.get('n_neighbors', 5, type=int),
            weights=request.args.get('weights', 'uniform'),
            metric=request.args.get('metric', 'minkowski')
        )
    elif model_type == 'Logistic Regression':
        return LogisticRegression(
            C=request.args.get('C', 1.0, type=float),
            penalty=request.args.get('penalty', 'l2'),
            solver=request.args.get('solver', 'liblinear'),
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'Naive Bayes':
        return GaussianNB(
            var_smoothing=request.args.get('var_smoothing', 1e-9, type=float)
        )
    elif model_type == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=request.args.get('n_estimators', 100, type=int),
            max_depth=request.args.get('max_depth', None, type=int),
            min_samples_split=request.args.get('min_samples_split', 2, type=int),
            criterion=request.args.get('criterion', 'gini'),
            random_state=42
        )
    elif model_type == 'Support Vector Machine':
        return make_pipeline(StandardScaler(), SVC(
            C=request.args.get('C', 1.0, type=float),
            kernel=request.args.get('kernel', 'rbf'),
            gamma=request.args.get('gamma', 'scale'),
            probability=True,
            random_state=42
        ))
    else:
        return None

# Endpoint: Confusion Matrix
@app.route('/confusion_matrix', methods=['GET'])
def display_confusion_matrix():
    try:
        user_id = request.args.get('user_id')  
        if not user_id:
            return jsonify({"status": "error", "message": "User ID is required."}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM uploaded_files WHERE user_id = %s ORDER BY uploaded_at DESC LIMIT 1",
                (user_id,)
            )
            file_record = cursor.fetchone()
        conn.close()

        if not file_record:
            return jsonify({"status": "error", "message": "No file found for this user."}), 400

        file_path = file_record['file_path']
        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": f"File not found: {file_path}"}), 400

        # Read dataset
        df = pd.read_csv(file_path)
        if df.empty:
            return jsonify({"status": "error", "message": "Dataset is empty."}), 400

        # Get target column from URL parameter
        target_column = get_target_column_from_meta(file_path)
        if target_column not in df.columns:
            return jsonify({
                "status": "error",
                "message": f"Target column '{target_column}' not found in the dataset."
            }), 400

        # Check for missing values
        if df.isnull().values.any():
            return jsonify({
                "status": "error",
                "message": "Dataset contains missing values. Please clean the data first through the 'clean data' option."
            }), 400

        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Failed to separate features and target: {str(e)}"
            }), 400

        # Check for categorical data
        if X.select_dtypes(include=['object']).shape[1] > 0:
            return jsonify({
                "status": "error",
                "message": "Dataset contains categorical data. Please convert the data first through the 'convert data' option."
            }), 400

        # Validate test size
        test_size = request.args.get('test_size', 0.3, type=float)
        if not (0 < test_size < 1):
            return jsonify({
                "status": "error",
                "message": "Test size must be between 0 and 1."
            }), 400

        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        # Save Training & Testing Data (fitur + label)
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Save Data Splitting
        train_data.to_csv("processed/split/train_data.csv", index=False)
        test_data.to_csv("processed/split/test_data.csv", index=False)      

        # Calculate number of training data
        train_data_size = len(X_train)

        model_type = request.args.get('model', 'Decision Tree')

        model = build_model(model_type, request)
        if model is None:
            return jsonify({"status": "error", "message": "Model yang dipilih tidak valid."}), 400

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Confusion matrix and classification report for training data
        cm_train = confusion_matrix(y_train, y_train_pred)
        accuracy_train = accuracy_score(y_train, y_train_pred)
        report_train = classification_report(y_train, y_train_pred, output_dict=True)
        class_metrics_train = {
            k: {
                "precision": report_train[k]["precision"],
                "recall": report_train[k]["recall"],
                "f1_score": report_train[k]["f1-score"],
                "support": report_train[k]["support"],
            }
            for k in report_train if k != 'accuracy'
        }

        # Confusion matrix and classification report for testing data
        cm_test = confusion_matrix(y_test, y_test_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report_test = classification_report(y_test, y_test_pred, output_dict=True)
        class_metrics_test = {
            k: {
                "precision": report_test[k]["precision"],
                "recall": report_test[k]["recall"],
                "f1_score": report_test[k]["f1-score"],
                "support": report_test[k]["support"],
            }
            for k in report_test if k != 'accuracy'
        }

        # Create confusion matrix image for training data
        fig_train, ax_train = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        ax_train.set_xlabel('Predicted')
        ax_train.set_ylabel('Actual')
        ax_train.set_title(f'{model_type}')
        buf_train = io.BytesIO()
        plt.savefig(buf_train, format='png')
        buf_train.seek(0)
        img_base64_train = base64.b64encode(buf_train.getvalue()).decode('utf-8')
        buf_train.close()
        plt.close(fig_train)

        # Create confusion matrix image
        fig_test, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_type}') 
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig_test)

        return jsonify({
            "status": "success",
            "model": model_type,
            "confusion_matrix_test": img_base64,
            "confusion_matrix_train": img_base64_train,
            "accuracy_test": accuracy_test,
            "accuracy_train": accuracy_train,
            "train_data_size": train_data_size,
            "class_metrics_test": class_metrics_test,
            "class_metrics_train": class_metrics_train,
        }), 200


    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An error occurred while generating the confusion matrix: {str(e)}"
        }), 500


# Endpoint: ROC Curve
@app.route('/roc_curve', methods=['GET'])
def roc_curve_plot():
    try:
        user_id = request.args.get('user_id')
        models_param = request.args.get('models', '')
        test_size = request.args.get('test_size', 0.3, type=float)

        if not user_id or not models_param:
            return jsonify({"status": "error", "message": "User ID and models parameter are required."}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM uploaded_files WHERE user_id = %s ORDER BY uploaded_at DESC LIMIT 1", (user_id,))
            file_record = cursor.fetchone()
        conn.close()

        if not file_record or not os.path.exists(file_record['file_path']):
            return jsonify({"status": "error", "message": "File not found."}), 400

        df = pd.read_csv(file_record['file_path'])
        if df.empty or df.isnull().values.any():
            return jsonify({"status": "error", "message": "Dataset empty or contains missing values."}), 400

        file_path = file_record['file_path']
        target_column = get_target_column_from_meta(file_path)

        if target_column not in df.columns:
            return jsonify({"status": "error", "message": f"Target column '{target_column}' not found."}), 400

        X = df.drop(columns=[target_column])
        y = df[target_column]

        if X.select_dtypes(include=['object']).shape[1] > 0:
            return jsonify({"status": "error", "message": "Dataset contains categorical data. Please convert it first."}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)

        y_test_bin = label_binarize(y_test, classes=y.unique())
        n_classes = y_test_bin.shape[1]

        model_names = [m.strip() for m in models_param.split(',')]

        plt.figure(figsize=(8, 6))
        for name in model_names:
            model = build_model(name, request)
            if model is None:
                print(f"Model not recognized: {name}")
                continue
            model.fit(X_train, y_train)
            if not hasattr(model, 'predict_proba'):
                continue
            y_score = model.predict_proba(X_test)

            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            mean_auc = sum(roc_auc.values()) / n_classes
            plt.plot(fpr[0], tpr[0], label=f"{name} (AUC = {mean_auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve Comparison")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='lower right')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        return jsonify({
            "status": "success",
            "roc_curve_combined": img_base64,
            "message": f"ROC curve for models: {', '.join(model_names)}"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Endpoint Prediction
@app.route('/predict', methods=['POST'])
def predict_disease():
    global label_encoder_target
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"status": "error", "message": "User ID is required."}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM uploaded_files WHERE user_id = %s ORDER BY uploaded_at DESC LIMIT 1",
                (user_id,)
            )
            file_record = cursor.fetchone()
        conn.close()

        if not file_record:
            return jsonify({"status": "error", "message": "No files found for this user."})

        file_path = file_record['file_path']
        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": f"File Not Found: {file_path}"})

        df = pd.read_csv(file_path)
        if df.empty:
            return jsonify({"status": "error", "message": "Empty dataset."})

        target_column = get_target_column_from_meta(file_path)
        if target_column not in df.columns:
            return jsonify({"status": "error", "message": f"Target column '{target_column}' not found in the dataset."})

        y = label_encoder_target.fit_transform(df[target_column])
        X = df.drop(columns=[target_column])
        X = X.apply(LabelEncoder().fit_transform) if X.select_dtypes(include=['object']).any().any() else X

        model_type = request.args.get('model', 'Decision Tree')

        model = build_model(model_type, request)
        if model is None:
            return jsonify({"status": "error", "message": "The selected model is not valid."}), 400

        model.fit(X, y)

        input_features = request.json.get('input_features')
        if not input_features:
            return jsonify({"status": "error", "message": "Input feature is required."})

        input_df = pd.DataFrame([input_features], columns=X.columns)
        prediction = model.predict(input_df)
        predicted_class = prediction[0]
        predicted_label = label_encoder_target.inverse_transform([predicted_class])[0]

        return jsonify({
            "status": "success",
            "prediction": predicted_label,
            "message": "The prediction was successfully."
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# Endpoint Clustering
@app.route('/clustering', methods=['GET'])
def clustering_endpoint():
    try:
        user_id = request.args.get('user_id')
        algorithm = request.args.get('algorithm', 'kmeans').lower()
        n_clusters = int(request.args.get('n_clusters', 3))
        eps = float(request.args.get('eps', 0.5))
        min_samples = int(request.args.get('min_samples', 5))

        if not user_id:
            return jsonify({'status': 'error', 'message': 'User ID is required.'}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM uploaded_files WHERE user_id = %s ORDER BY uploaded_at DESC LIMIT 1",
                (user_id,))
            file_record = cursor.fetchone()
        conn.close()

        if not file_record:
            return jsonify({'status': 'error', 'message': 'No uploaded file found for this user.'}), 404

        file_path = file_record['file_path']
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': f'File not found: {file_path}'}), 404

        df = pd.read_csv(file_path)
        if 'Level' in df.columns:
            df = df.drop(columns=['Level'])

        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            return jsonify({'status': 'error', 'message': 'Dataset must contain at least 2 numerical columns.'}), 400

        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df)

        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = model.fit_predict(X)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = model.fit_predict(X)
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            clusters = model.fit(X).predict(X)
        else:
            return jsonify({'status': 'error', 'message': f'Unsupported algorithm: {algorithm}'}), 400

        df['Level'] = clusters
        df.to_csv(file_path, index=False)

        # Visualization
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=X[:, 0], y=X[:, 1],
            hue=clusters,
            palette='Set2',
            s=60
        )
        plt.title(f'{algorithm.upper()} Clustering')
        plt.xlabel(numeric_df.columns[0])
        plt.ylabel(numeric_df.columns[1])
        plt.legend(title='Cluster')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        return jsonify({
            'status': 'success',
            'algorithm': algorithm,
            'image': img_base64,
            'data_with_clusters': df.to_dict(orient='records'),
            'message': f'Clustering completed using {algorithm.upper()}'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Main
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
