from app import create_app

if __name__ == "__main__":
    test_app = create_app()
    test_app.debug = True
    test_app.run(host="0.0.0.0", port=1000)

