import os

def touch(filepath):
    # Create empty file if it does not exist
    if not os.path.exists(filepath):
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Create the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("")  # empty
    else:
        # If it exists and is a file, do nothing; if it's a directory, skip
        if os.path.isdir(filepath):
            return

def main():
    base_path = r"C:\Users\yassi\Documents\projects\pathmnist XAI"

    # Directories to create
    dirs = [
        "",  # base
        "weights",
        "models",
        "utils",
        "templates",
        os.path.join("templates", "partials"),
        "static",
        os.path.join("static", "css"),
        os.path.join("static", "js"),
        os.path.join("static", "images"),
        os.path.join("static", "overlays"),
        os.path.join("static", "data"),
    ]

    # Files to create (empty)
    files = [
        "app.py",
        "config.py",
        "requirements.txt",
        "README.md",

        os.path.join("weights", "best_pathmnist_resnet18.pth"),  # placeholder empty file
        os.path.join("models", "model_loader.py"),

        os.path.join("utils", "__init__.py"),
        os.path.join("utils", "preprocessing.py"),
        os.path.join("utils", "xai_methods.py"),
        os.path.join("utils", "metrics.py"),

        os.path.join("templates", "base.html"),
        os.path.join("templates", "index.html"),
        os.path.join("templates", "partials", "prediction.html"),
        os.path.join("templates", "partials", "explanation.html"),
        os.path.join("templates", "partials", "metrics.html"),

        os.path.join("static", "css", "custom.css"),
        os.path.join("static", "js", "main.js"),
        # images/ and overlays/ are directories; no default files needed
        os.path.join("static", "data", "metrics.json"),
        os.path.join("static", "data", "confusion_matrix.json"),
        os.path.join("static", "data", "class_samples.json"),
    ]

    # Create directories
    for d in dirs:
        path = os.path.join(base_path, d)
        os.makedirs(path, exist_ok=True)

    # Create empty files
    for fpath in files:
        full = os.path.join(base_path, fpath)
        touch(full)

    print(f"Project skeleton created at: {base_path}")

if __name__ == "__main__":
    main()
