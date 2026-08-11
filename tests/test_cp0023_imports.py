def test_window_imports():
    from linecaller.annotation.window import AnnotationStudioWindow
    assert AnnotationStudioWindow is not None
def test_tool_imports():
    import tools.annotation_studio as app
    assert callable(app.main)
