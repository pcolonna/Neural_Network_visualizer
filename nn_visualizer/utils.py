import io


def summarize(model):
    """ Returns the summary for the model to streamlit"""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string
