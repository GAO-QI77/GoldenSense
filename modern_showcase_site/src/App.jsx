import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu, X, ChevronRight, Star, Users, Shield, 
  Mail, Phone, MapPin, Github, Twitter, Linkedin,
  Layers, Zap, Globe, Cpu
} from 'lucide-react';

// --- Components ---

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { name: 'Home', href: '#home' },
    { name: 'Services', href: '#services' },
    { name: 'About', href: '#about' },
    { name: 'Contact', href: '#contact' },
  ];

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${scrolled ? 'glass py-4' : 'bg-transparent py-6'}`}>
      <div className="container mx-auto px-6 flex justify-between items-center">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-2xl font-bold gradient-text cursor-pointer"
        >
          MODERN.LAB
        </motion.div>

        {/* Desktop Nav */}
        <div className="hidden md:flex space-x-8 items-center">
          {navLinks.map((link) => (
            <a 
              key={link.name} 
              href={link.href}
              className="text-gray-300 hover:text-white transition-colors text-sm font-medium"
            >
              {link.name}
            </a>
          ))}
          <button className="bg-primary hover:bg-blue-600 px-5 py-2 rounded-full text-white text-sm font-semibold transition-all">
            Get Started
          </button>
        </div>

        {/* Mobile Toggle */}
        <div className="md:hidden">
          <button onClick={() => setIsOpen(!isOpen)} className="text-white">
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden glass border-t border-white/10 overflow-hidden"
          >
            <div className="flex flex-col p-6 space-y-4">
              {navLinks.map((link) => (
                <a 
                  key={link.name} 
                  href={link.href}
                  onClick={() => setIsOpen(false)}
                  className="text-gray-300 hover:text-white text-lg"
                >
                  {link.name}
                </a>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

const Hero = () => (
  <section id="home" className="min-h-screen flex items-center pt-20 relative overflow-hidden">
    <div className="absolute top-0 left-0 w-full h-full -z-10">
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-[120px]" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px]" />
    </div>

    <div className="container mx-auto px-6 grid md:grid-cols-2 gap-12 items-center">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className="inline-flex items-center px-4 py-2 glass rounded-full text-primary text-xs font-bold mb-6">
          <Zap size={14} className="mr-2" /> NEXT GEN TECHNOLOGY
        </div>
        <h1 className="text-5xl md:text-7xl font-bold leading-tight mb-6">
          Elevate Your <span className="gradient-text">Digital Vision</span> To Reality
        </h1>
        <p className="text-gray-400 text-lg mb-8 max-w-lg">
          We build cutting-edge digital experiences using the latest web technologies. 
          Scalable, secure, and beautiful by design.
        </p>
        <div className="flex space-x-4">
          <button className="bg-primary hover:bg-blue-600 px-8 py-4 rounded-xl text-white font-bold transition-all flex items-center">
            Explore Projects <ChevronRight size={20} className="ml-2" />
          </button>
          <button className="glass hover:bg-white/10 px-8 py-4 rounded-xl text-white font-bold transition-all">
            Learn More
          </button>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="relative"
      >
        <div className="glass p-4 rounded-[2rem] relative z-10">
          <img 
            src="https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=2426&auto=format&fit=crop" 
            alt="Digital Solution" 
            className="rounded-2xl shadow-2xl"
          />
        </div>
        <div className="absolute -top-10 -right-10 glass p-6 rounded-2xl animate-bounce hidden md:block">
          <Star className="text-accent" fill="currentColor" />
        </div>
      </motion.div>
    </div>
  </section>
);

const Services = () => {
  const services = [
    { icon: <Globe />, title: 'Web Development', desc: 'Modern frameworks like React, Next.js, and Vue.' },
    { icon: <Cpu />, title: 'AI Integration', desc: 'Smart automation and machine learning solutions.' },
    { icon: <Layers />, title: 'UI/UX Design', desc: 'User-centric designs that convert and delight.' },
    { icon: <Shield />, title: 'Cyber Security', desc: 'Enterprise-grade protection for your digital assets.' },
  ];

  return (
    <section id="services" className="py-24 bg-slate-900/50">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">Our <span className="gradient-text">Premium Services</span></h2>
          <p className="text-gray-400 max-w-2xl mx-auto">We provide a wide range of services to help you scale your business in the digital age.</p>
        </div>

        <div className="grid md:grid-cols-4 gap-8">
          {services.map((s, idx) => (
            <motion.div
              key={idx}
              whileHover={{ y: -10 }}
              className="glass p-8 rounded-3xl hover:border-primary/50 transition-colors cursor-pointer"
            >
              <div className="w-12 h-12 bg-primary/20 rounded-xl flex items-center justify-center text-primary mb-6">
                {s.icon}
              </div>
              <h3 className="text-xl font-bold mb-4">{s.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{s.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

const About = () => (
  <section id="about" className="py-24">
    <div className="container mx-auto px-6 grid md:grid-cols-2 gap-16 items-center">
      <div className="relative order-2 md:order-1">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-4 mt-8">
            <div className="glass p-6 rounded-3xl">
              <h4 className="text-3xl font-bold text-primary mb-2">150+</h4>
              <p className="text-gray-400 text-xs">Projects Completed</p>
            </div>
            <div className="glass p-6 rounded-3xl">
              <h4 className="text-3xl font-bold text-accent mb-2">98%</h4>
              <p className="text-gray-400 text-xs">Client Satisfaction</p>
            </div>
          </div>
          <div className="space-y-4">
            <div className="glass p-6 rounded-3xl">
              <h4 className="text-3xl font-bold text-purple-400 mb-2">12+</h4>
              <p className="text-gray-400 text-xs">Expert Developers</p>
            </div>
            <div className="glass p-6 rounded-3xl">
              <h4 className="text-3xl font-bold text-green-400 mb-2">24/7</h4>
              <p className="text-gray-400 text-xs">Support Available</p>
            </div>
          </div>
        </div>
      </div>

      <div className="order-1 md:order-2">
        <h2 className="text-4xl font-bold mb-6">We Are The <span className="gradient-text">Future Of Web</span> Development</h2>
        <p className="text-gray-400 mb-6 leading-relaxed">
          Founded in 2020, MODERN.LAB has been at the forefront of digital transformation. 
          We believe that technology should be accessible, efficient, and exceptionally designed.
        </p>
        <ul className="space-y-4 mb-8">
          {[
            { icon: <Users size={18} />, text: 'Dedicated Project Managers' },
            { icon: <Zap size={18} />, text: 'Agile Development Process' },
            { icon: <Shield size={18} />, text: 'Secure & Scalable Infrastructure' }
          ].map((item, i) => (
            <li key={i} className="flex items-center text-gray-300">
              <span className="text-primary mr-3">{item.icon}</span> {item.text}
            </li>
          ))}
        </ul>
        <button className="bg-white text-black hover:bg-gray-200 px-8 py-3 rounded-full font-bold transition-all">
          Meet The Team
        </button>
      </div>
    </div>
  </section>
);

const Contact = () => (
  <section id="contact" className="py-24 bg-slate-900/50">
    <div className="container mx-auto px-6">
      <div className="glass p-12 rounded-[3rem] grid md:grid-cols-2 gap-12">
        <div>
          <h2 className="text-4xl font-bold mb-6">Ready To <span className="gradient-text">Start A Project?</span></h2>
          <p className="text-gray-400 mb-8">Get in touch with us today and let's discuss how we can help your business grow.</p>
          
          <div className="space-y-6">
            <div className="flex items-center">
              <div className="w-10 h-10 glass rounded-full flex items-center justify-center text-primary mr-4">
                <Mail size={18} />
              </div>
              <span className="text-gray-300">hello@modernlab.tech</span>
            </div>
            <div className="flex items-center">
              <div className="w-10 h-10 glass rounded-full flex items-center justify-center text-primary mr-4">
                <Phone size={18} />
              </div>
              <span className="text-gray-300">+1 (555) 123-4567</span>
            </div>
            <div className="flex items-center">
              <div className="w-10 h-10 glass rounded-full flex items-center justify-center text-primary mr-4">
                <MapPin size={18} />
              </div>
              <span className="text-gray-300">Silicon Valley, CA, USA</span>
            </div>
          </div>
        </div>

        <form className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <input type="text" placeholder="First Name" className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary transition-all" />
            <input type="text" placeholder="Last Name" className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary transition-all" />
          </div>
          <input type="email" placeholder="Email Address" className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary transition-all" />
          <textarea placeholder="Your Message" rows="4" className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-primary transition-all"></textarea>
          <button className="w-full bg-primary hover:bg-blue-600 py-4 rounded-xl text-white font-bold transition-all">
            Send Message
          </button>
        </form>
      </div>
    </div>
  </section>
);

const Footer = () => (
  <footer className="py-12 border-t border-white/10">
    <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center">
      <div className="text-xl font-bold gradient-text mb-6 md:mb-0">MODERN.LAB</div>
      
      <div className="flex space-x-8 mb-6 md:mb-0 text-sm text-gray-400">
        <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
        <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
        <a href="#" className="hover:text-white transition-colors">Cookie Policy</a>
      </div>

      <div className="flex space-x-4">
        {[<Twitter size={20} />, <Linkedin size={20} />, <Github size={20} />].map((icon, i) => (
          <a key={i} href="#" className="w-10 h-10 glass rounded-full flex items-center justify-center text-gray-400 hover:text-white hover:bg-primary/20 transition-all">
            {icon}
          </a>
        ))}
      </div>
    </div>
    <div className="text-center text-gray-500 text-xs mt-8">
      © 2024 MODERN.LAB Open Source Project. Licensed under MIT.
    </div>
  </footer>
);

export default function App() {
  return (
    <div className="bg-[#0f172a] text-white selection:bg-primary/30">
      <Navbar />
      <Hero />
      <Services />
      <About />
      <Contact />
      <Footer />
    </div>
  );
}
