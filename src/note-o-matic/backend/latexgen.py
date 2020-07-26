from pylatex import Document, Section, Subsection, Command, Figure, Itemize
from pylatex.utils import italic, NoEscape
import os


def fill_document(doc):
    """Add a section, a subsection and some text to the document.

    :param doc: the document
    :type doc: :class:`pylatex.document.Document` instance
    """
    with doc.create(Section('Details of Original Notes')):
        doc.append('Word count: ')
        doc.append(italic('something'))

        with doc.create(Subsection('Key Words')):
            with doc.create(Itemize()) as itemize:
                itemize.add_item("the first item")
                itemize.add_item("the second item")
                itemize.add_item("the third etc")
                # you can append to existing items
                itemize.append(Command("ldots"))


if __name__ == '__main__':
    # Basic document
    doc = Document('basic')
    fill_document(doc)

    # Document with `\maketitle` command activated
    doc = Document()
    image_filename = os.path.join(os.path.dirname(__file__), 'final.png')

    
    with doc.create(Figure(position='h!')) as logo:
        logo.add_image(image_filename, width='300px')
    doc.preamble.append(Command('title', 'note-o-matic Summary'))
    doc.preamble.append(Command('author', 'Generated using note-o-matic'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))


    fill_document(doc)

    
    # Add stuff to the document
    with doc.create(Section('Your notes - reworked.')):
        doc.append('uwu')

    doc.generate_pdf('note-o-maticSummary', clean_tex=False, compiler='/Library/TeX/texbin/pdflatex')
    tex = doc.dumps()  # The document as string in LaTeX syntax
