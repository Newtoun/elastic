import React from "react";

const ResultadosPesquisa = ({ results }) => {
    return (
      <ul>
        {results.map((result, index) => (
          <li key={index}>
            <h3>{result.title}</h3>
            <p>{result.description}</p>
          </li>
        ))}
      </ul>
    );
};

export default ResultadosPesquisa