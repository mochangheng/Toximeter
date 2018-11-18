# Toximeter

This API is built to faciliate the analysis of text for Python apps.

## How to use it?

Clone this repository and copy the toximeter.py file into the folder of your Python app.

```
git clone https://github.com/mochangheng/Toximeter.git
```

Import the package
```python
from toximeter import Toximeter
```

Analyze text
```python
analyzer = Toximeter()
result = analyzer.analyze('Hello World.')
```

A sample result: `{"toxicity": 1.0, "sentiment": 67.0, "tone": {"tone": "Neutral", "score": 100}}`
