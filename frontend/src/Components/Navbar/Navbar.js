import React, { useState, useEffect } from 'react'
import './Navbar.css'
import { Link } from 'react-scroll';
//import logo from '../../assets/icon.png'
import logo1 from '../Images/logo.png'


const Navbar = () => {

  const [sticky,setSticky] = useState(false);

  useEffect(()=>{
    window.addEventListener('scroll',()=>{
      window.scrollY > 500 ? setSticky(true) : setSticky(false);
    })
  },[]);
  return (
    <nav className={`container ${sticky? 'dark-nav' : ''}`}>
        <img src={logo1} alt='Website Logo' className='logo' />
        <div className='logotxt'>RE-SEARCH BOT</div>
        <ul>
            <li><Link to='homerouter' smooth={true} offset={0} duration={1500}>Home</Link></li>
            <li>About us</li>
            <li><button className='btn'>Pricing</button></li>
        </ul>
    </nav>
  )
}

export default Navbar;
