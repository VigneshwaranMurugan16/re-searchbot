import React from 'react'
import './Home.css'
import arrow from '../Images/arrow.png'
import { delay, motion } from "framer-motion"

import { Link } from 'react-scroll';

const Home = () => {
  return (
    <div className='home container'>
      <div className='home-text'>
      <motion.h1 
      whileInView={{opacity: 1,y:0}}
      initial={{opacity:0, y:15}}
      transition={{duration:1}}>A Complete Analysis Bot For Business Applications & Personal Use.</motion.h1>
      
      <motion.p
      whileInView={{opacity: 1,y:0}}
      initial={{opacity:0, y:20}}
      transition={{duration:1}}>Use sample version of the product to analysis documents and answer effectively without hallucinations and inconsistencies.</motion.p>
      <button className='btn'><Link to='uploadrouter' smooth={true} offset={-100} duration={1500}>Get Started</Link><img src={arrow} alt=''/></button>
      </div>
      

    </div>
  )
}

export default Home