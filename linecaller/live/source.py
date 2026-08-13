def parse_video_source(value):
    """
    Accept:
      "0" -> camera index 0
      "1" -> camera index 1
      any other string -> file/stream URL/path
    """
    if isinstance(value, int):
        return value

    text = str(value).strip()

    if text.isdigit():
        return int(text)

    return text
