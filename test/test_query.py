from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_query_yields_10_results():
    response = client.get("/query?query=Elon Musk anuncia novo projeto de exploração espacial")
    json_response = response.json()
    
    assert response.status_code == 200
    assert len(json_response["results"]) == 10
    assert json_response["message"] == "OK"

def test_query_yields_few_results():
    response = client.get("/query?query=Brasil vence a Copa do Mundo")
    json_response = response.json()
    
    assert response.status_code == 200
    assert 1 < len(json_response["results"]) < 10
    assert json_response["message"] == "OK"

def test_query_yields_non_obvious_results():
    """A relevância aqui pode não ser imediatamente aparente, já que "crescimento verde" pode se referir tanto a políticas 
    ambientais quanto a iniciativas econômicas sustentáveis. O resultado pode ser não óbvio se ele priorizar notícias sobre
    economia em vez de ecologia, por exemplo."""
    response = client.get("/query?query=crescimento verde")
    json_response = response.json()
    
    # TODO: add assert to verify non obvious results
    assert response.status_code == 200
    assert len(json_response["results"]) > 0
    assert json_response["message"] == "OK"
