#set document(title: "ChartLibre — User Manual", author: "ChartLibre")
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

#let version = "0.1.0"

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(0.4em)
  block(text(size: 20pt, weight: "bold", it.body))
  v(0.6em)
  line(length: 100%, stroke: 0.6pt + gray)
  v(0.6em)
}

#show heading.where(level: 2): it => {
  v(0.6em)
  block(text(size: 13.5pt, weight: "bold", it.body))
  v(0.2em)
}

#show heading.where(level: 3): it => {
  block(text(size: 11pt, weight: "bold", style: "italic", it.body))
}


#import "sections/helpers.typ": *

// ----------------------------------------------------------------------
// Cover
// ----------------------------------------------------------------------
#align(center)[
  #v(4cm)
  #text(size: 30pt, weight: "bold")[ChartLibre]
  #v(0.3cm)
  #text(size: 15pt, style: "italic")[User Manual]
  #v(1.5cm)
  #text(size: 11pt, fill: gray)[Application version #version]
  #v(0.3cm)
  #text(size: 10pt, fill: gray)[A desktop application for exploring data and building scientific charts]
]

#pagebreak()
#outline(title: "Contents", indent: auto)

#include "sections/intro.typ"
#include "sections/start.typ"
#include "sections/window.typ"
#include "sections/data.typ"
#include "sections/charts.typ"
#include "sections/operations.typ"
#include "sections/statistics.typ"
#include "sections/output.typ"
#include "sections/settings.typ"
#include "sections/advanced.typ"
