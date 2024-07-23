import React, { useState } from 'react'
import './styles.css'
import BarraPesquisa from "./componentes/barraDePesquisa";
import ResultadosPesquisa from "./componentes/resultadosDaPesquisa"

function App() {
  const [results, setResult] = useState([])

  const pesquisa = async (query) => {
    console.log("Pesquisando:",query)
    try {
      const response = await fetch(`http://localhost:5000/search?q=${query}`)
      console.log(response.status)
      const data = await response.json();
      console.log('resultados:', data)
      setResult(data)
    } catch(error) {
      console.error('Não foi possível fazer pesquisa, ERROR:',error)
    }
  }

  return (
    <div style={{ marginTop: '5rem' }}>
      <BarraPesquisa onSearch={pesquisa}/>
      <ResultadosPesquisa results={results}/>
    </div>
  ); 
}

export default App;
