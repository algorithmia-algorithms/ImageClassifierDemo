from . import ImageClassifierDemo

def test_ImageClassifierDemo():
    assert ImageClassifierDemo.apply("Jane") == "hello Jane"
