import React, { useState } from 'react';

const BarraPesquisa = ({ onSearch }) => {
  const [query, setQuery] = useState('');

  const pesquisa = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <div className='barra-pesquisa'>
      <form onSubmit={pesquisa}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search..."
        />
        <button type="submit">Pesquisar</button>
      </form>
    </div>
  );
};

export default BarraPesquisa;