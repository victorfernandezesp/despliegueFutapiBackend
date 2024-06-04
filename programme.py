from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np


app = Flask(__name__)
CORS(app)

# Cargar el scaler y el modelo KMeans
scaler = joblib.load('./modelos/clustering/scaler.pkl')
kmeans = joblib.load('./modelos/clustering/kmeans_model.pkl')

# Cargar los datos de jugadores
datos = pd.read_csv("conjunto_prueba.csv")

# Asegurarse de que la columna 'Cluster' esté en los datos
stats = ['PasTotCmp%', 'Goals', 'Shots', 'Assists', 'ScaDrib', 'TklWon', 'GcaDrib', 'Int', 'PasTotCmp', 'AerWon']
datos_scaled = scaler.transform(datos[stats])
datos['Cluster'] = kmeans.predict(datos_scaled)

@app.route('/getPlayers')
def devolverJugadores():
    json_file_path = 'db.json'  
    return send_file(json_file_path, mimetype='application/json')

@app.route('/getPlayerByRk/<int:rk>', methods=['GET'])
def player_by_rk(rk):
    jugador_datos = datos[datos['Rk'] == rk]

    if jugador_datos.empty:
        return jsonify({'mensaje': 'Jugador no encontrado en los datos.'}), 404
    
    return jugador_datos.to_json(orient='records'), 200

clasificationModel91 = joblib.load("./modelos/clasificacion/ClasificationModel91.pkl")

@app.route('/predictPosition', methods=['POST'])
def adivinaPosicion():
    datos_jugador = request.json

    datos_jugador = {k: v for k, v in datos_jugador.items() if isinstance(v, (int, float)) and k != 'Pos'}

    datos_np = np.array([list(datos_jugador.values())])

    prediccion = clasificationModel91.predict(datos_np)

    posiciones = ["Portero", "Defensa", "Mediocampista", "Delantero"]
    posicion = posiciones[prediccion[0]]

    return jsonify({'posicion': posicion}), 200

predictionXgoals = joblib.load('./modelos/prediccion/PredictionxG.pkl')

@app.route('/expectedGoals', methods=['POST'])
def predecir_goles():
   # Obtener los datos JSON del cuerpo de la solicitud
    datos_jugador = request.json
    
    # Verificar si las columnas a eliminar están presentes en los datos JSON
    columnas_a_eliminar = ['Player', 'Nation', 'Squad', 'Comp', 'Goals']
    columnas_presentes = [col for col in columnas_a_eliminar if col in datos_jugador]

    # Si todas las columnas a eliminar están presentes en los datos JSON
    if len(columnas_presentes) == len(columnas_a_eliminar):
        # Convertir los datos del jugador en un DataFrame de pandas
        datos_jugador_df = pd.DataFrame([datos_jugador])

        # Eliminar solo las columnas existentes en el DataFrame
        datos_jugador_df.drop(columns=columnas_presentes, inplace=True)

        # Utilizar el modelo cargado para predecir las GCA del nuevo jugador
        prediccion = predictionXgoals.predict(datos_jugador_df)

        if prediccion[0] < 0:
            prediccion[0] = 0
        prediccion[0] *= 1.75;
        # Devolver la predicción como una respuesta JSON
        return jsonify({"expectedGoals": prediccion.tolist()})
    else:
        return jsonify({"error": "Las columnas necesarias no están presentes en los datos JSON."}), 400


predictionXAssists = joblib.load('./modelos/prediccion/PredictionxA.pkl')

@app.route('/expectedAssists', methods=['POST'])
def predecir_asistencias():
     # Obtener los datos JSON del cuerpo de la solicitud
    datos_jugador = request.json
    
    # Verificar si las columnas a eliminar están presentes en los datos JSON
    columnas_a_eliminar = ['Player', 'Nation', 'Squad', 'Comp', 'Assists']
    columnas_presentes = [col for col in columnas_a_eliminar if col in datos_jugador]

    # Si todas las columnas a eliminar están presentes en los datos JSON
    if len(columnas_presentes) == len(columnas_a_eliminar):
        # Convertir los datos del jugador en un DataFrame de pandas
        datos_jugador_df = pd.DataFrame([datos_jugador])

        # Eliminar solo las columnas existentes en el DataFrame
        datos_jugador_df.drop(columns=columnas_presentes, inplace=True)

        # Utilizar el modelo cargado para predecir las GCA del nuevo jugador
        prediccion = predictionXAssists.predict(datos_jugador_df)

        if prediccion[0] < 0:
            prediccion[0] = 0
        prediccion[0] *= 1.5;
        # Devolver la predicción como una respuesta JSON
        return jsonify({"expectedAssists": prediccion.tolist()})
    else:
        return jsonify({"error": "Las columnas necesarias no están presentes en los datos JSON."}), 400

predicTionTkl = joblib.load('./modelos/prediccion/PrediccionTkl99.pkl')

@app.route('/expectedTackles', methods=['POST'])
def predecir_tackles():
    # Obtener los datos JSON del cuerpo de la solicitud
    datos_jugador = request.json

    # Convertir los datos del jugador en un DataFrame de pandas
    datos_jugador_df = pd.DataFrame([datos_jugador])

    # Verificar si las columnas a eliminar están presentes en el DataFrame
    columnas_a_eliminar = ['Player', 'Nation', 'Squad', 'Comp', 'Tkl']
    columnas_existentes = set(datos_jugador_df.columns)
    columnas_a_eliminar = [col for col in columnas_a_eliminar if col in columnas_existentes]

    # Eliminar solo las columnas existentes
    if columnas_a_eliminar:
        datos_jugador_df.drop(columns=columnas_a_eliminar, inplace=True)

    # Utilizar el modelo cargado para predecir las tackles del nuevo jugador
    prediccion = predicTionTkl.predict(datos_jugador_df)

    # Devolver la predicción como una respuesta JSON
    return jsonify({"expectedTackles": prediccion.tolist()})

# Cargar el modelo entrenado
predictToSuc = joblib.load('./modelos/prediccion/DefDribbledENTR_85_011.pkl')

@app.route('/expectedDribbles', methods=['POST'])
def predecir_def_dribbled_success():
    # Obtener los datos JSON del cuerpo de la solicitud
    datos_jugador = request.json

    datos_jugador_df = pd.DataFrame([datos_jugador])

    # Verificar si las columnas a eliminar están presentes en el DataFrame
    columnas_a_eliminar = ['Player', 'Nation', 'Squad', 'Comp', 'ToSuc']
    columnas_existentes = set(datos_jugador_df.columns)
    columnas_a_eliminar = [col for col in columnas_a_eliminar if col in columnas_existentes]

# Eliminar solo las columnas existentes
    if columnas_a_eliminar:
        datos_jugador_df.drop(columns=columnas_a_eliminar, inplace=True)

    # Utilizar el modelo cargado para predecir las tackles del nuevo jugador
    prediccion = predictToSuc.predict(datos_jugador_df)

    # Si la predicción es menor que 0, devolver 0
    if prediccion[0] < 0:
        prediccion[0] = 0

    # Devolver la predicción como una respuesta JSON
    return jsonify({"expectedDribbles": prediccion.tolist()})

@app.route('/searchSimilarPlayers', methods=['POST'])
def buscar_jugadores_similares():
    # Obtener los datos JSON del cuerpo de la solicitud
    datos_jugador = request.json
    
    # Verificar si se proporcionaron datos válidos
    if not datos_jugador:
        return jsonify({'mensaje': 'No se han proporcionado datos válidos.'}), 400
    
    # Convertir los datos del jugador en un DataFrame de pandas
    jugador_datos_df = pd.DataFrame([datos_jugador])

    # Seleccionar características del jugador
    stats = ['PasTotCmp%', 'Goals', 'Shots', 'Assists', 'ScaDrib', 'TklWon', 'GcaDrib', 'Int', 'PasTotCmp', 'AerWon']
    jugador_datos = jugador_datos_df[stats]

    # Escalar los datos del jugador
    jugador_datos_scaled = scaler.transform(jugador_datos)

    # Obtener el clúster del jugador
    jugador_cluster = kmeans.predict(jugador_datos_scaled)[0]
    
    # Encontrar todos los jugadores en el mismo clúster
    jugadores_similares = datos[datos['Cluster'] == jugador_cluster]['Player'].tolist()
    
    return jsonify({'searchSimilarPlayers': jugadores_similares})

predictGCA = joblib.load('./modelos/prediccion/PrediccionGCA.pkl')

@app.route('/expectedGCA', methods=['POST']) 
def buscar_jugadores_gca():
    # Obtener los datos JSON del cuerpo de la solicitud
    datos_jugador = request.json
    
    # Verificar si las columnas a eliminar están presentes en los datos JSON
    columnas_a_eliminar = ['Player', 'Nation', 'Squad', 'Comp', 'GCA']
    columnas_presentes = [col for col in columnas_a_eliminar if col in datos_jugador]

    # Si todas las columnas a eliminar están presentes en los datos JSON
    if len(columnas_presentes) == len(columnas_a_eliminar):
        # Convertir los datos del jugador en un DataFrame de pandas
        datos_jugador_df = pd.DataFrame([datos_jugador])

        # Eliminar solo las columnas existentes en el DataFrame
        datos_jugador_df.drop(columns=columnas_presentes, inplace=True)

        # Utilizar el modelo cargado para predecir las GCA del nuevo jugador
        prediccion = predictGCA.predict(datos_jugador_df)

        if prediccion[0] < 0:
            prediccion[0] = 0
        # Devolver la predicción como una respuesta JSON
        return jsonify({"expectedGCA": prediccion.tolist()})
    else:
        return jsonify({"error": "Las columnas necesarias no están presentes en los datos JSON."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)